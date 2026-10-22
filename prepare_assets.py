"""
One-off asset preparation for PaintRobotSimulation.py

Reads the two original STL files and writes the baked, ready-to-load meshes
into ./assets :
  Incredible_Jaban_m.stl            car, metres, centred, nose towards +Y
  MPX3500_base.stl ... MPX3500_T.stl  one mesh per robot link, expressed in
                                      that link's own DH frame at q = 0

Usage:  python prepare_assets.py  <folder containing the two original .stl files>
"""
import os, sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

# ---- CAD-derived DH (metres) -- must match PaintRobotSimulation.py ----------
D1, A1, A2, A3, D4, D6 = 0.795, 0.0, 1.264, -0.056, 1.442, 0.136
SCALE = 1.0 / 25400.0     # the MPX3500 STL is in 'inch-scaled' units: 25.4 units = 1 mm
TARGET_CELL = 0.006       # vertex-clustering cell (m) used to decimate the heavy shells


def load_stl(path):
    with open(path, 'rb') as f:
        f.read(80)
        n = int(np.frombuffer(f.read(4), dtype='<u4')[0])
        dt = np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
        d = np.frombuffer(f.read(n * 50), dtype=dt)
    return d['v'].astype(float)


def save_stl(path, tris):
    tris = np.asarray(tris, float)
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    l = np.linalg.norm(nrm, axis=1, keepdims=True); l[l == 0] = 1; nrm /= l
    dt = np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    d = np.zeros(len(tris), dtype=dt); d['n'] = nrm; d['v'] = tris
    with open(path, 'wb') as f:
        f.write(b'baked for PaintRobotSimulation'.ljust(80, b' '))
        f.write(np.uint32(len(tris)).tobytes()); f.write(d.tobytes())


def dh_A(theta, d, a, alpha):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st*ca, st*sa, a*ct], [st, ct*ca, -ct*sa, a*st],
                     [0, sa, ca, d], [0, 0, 0, 1]])


def zero_pose_frames():
    pi = np.pi
    P = [(0 - 0,        D1, A1, -pi/2), (-pi/2, 0, A2, 0), (0, 0, A3, -pi/2),
         (0, D4, 0, pi/2), (0, 0, 0, -pi/2), (0, D6, 0, 0)]
    T = np.eye(4); F = [T.copy()]
    for th, d, a, al in P:
        T = T @ dh_A(th, d, a, al); F.append(T.copy())
    return F


def split_shells(tris):
    P = tris.reshape(-1, 3)
    u, inv = np.unique(np.round(P, 2), axis=0, return_inverse=True)
    tri = inv.reshape(-1, 3)
    r = np.concatenate([tri[:, 0], tri[:, 1]]); c = np.concatenate([tri[:, 1], tri[:, 2]])
    G = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(len(u), len(u)))
    nc, lab = connected_components(G, directed=False)
    return nc, lab[tri[:, 0]]


def decimate(tris, cell):
    P = tris.reshape(-1, 3)
    key = np.floor(P / cell).astype(np.int64)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    cent = np.zeros((len(uk), 3)); cnt = np.bincount(inv, minlength=len(uk))
    for k in range(3):
        cent[:, k] = np.bincount(inv, weights=P[:, k], minlength=len(uk)) / cnt
    t = inv.reshape(-1, 3)
    ok = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
    t = t[ok]
    s = np.sort(t, axis=1)
    _, first = np.unique(s, axis=0, return_index=True)
    return cent[t[first]]


def main(src, out):
    os.makedirs(out, exist_ok=True)
    # ---------------- car ----------------
    v = load_stl(os.path.join(src, 'Incredible_Jaban.stl')) * 0.001
    P = v.reshape(-1, 3)
    v[:, :, 0] -= (P[:, 0].min() + P[:, 0].max()) / 2
    v[:, :, 1] -= (P[:, 1].min() + P[:, 1].max()) / 2
    v[:, :, 2] -= P[:, 2].min()
    v[:, :, 0] *= -1; v[:, :, 1] *= -1          # 180 deg about Z: nose -> +Y
    save_stl(os.path.join(out, 'Incredible_Jaban_m.stl'), v)

    # ---------------- robot ----------------
    tris = load_stl(os.path.join(src, 'ImageToStl_com_MPX3500-C10X_3D.stl')) * SCALE
    nc, tl = split_shells(tris * 25400.0)
    cnt = np.bincount(tl, minlength=nc)
    # identify shells by their position in the CAD pose (mesh frame: Y up, arm along +X)
    info = []
    for k in range(nc):
        Q = tris[tl == k].reshape(-1, 3)
        info.append((k, cnt[k], Q.min(0), Q.max(0)))
    # mesh (x, y_up, z) -> robot base frame (x, -z, y_up)
    to_world = lambda Q: np.column_stack([Q[:, 0], -Q[:, 2], Q[:, 1]])
    frames = zero_pose_frames()

    # Identify the 8 shells of the supplied STL from their size / position
    # (metres, mesh frame, arm lying in the q = 0 pose along +X, Y up).
    ids = {}
    for k, c, lo, hi in info:
        if c < 3000:                      ids['switch'] = k     # small box on the base
        elif hi[0] > 1.55:                ids['T'] = k          # flange (tip at x = 1.578 m)
        elif c < 30000 and c > 20000:     ids['B'] = k          # wrist bend housing
        elif c < 30000:                   ids['R'] = k          # forearm roll housing
        elif lo[1] < 0.10:                ids['base'] = k       # sits on the floor
        elif lo[1] < 0.50:                ids['S'] = k          # swivel housing
        elif lo[1] < 1.00:                ids['L'] = k          # lower arm
        else:                             ids['U'] = k          # upper arm / forearm housing
    assert len(ids) == 8, f'could not identify all shells: {ids}'
    print('shell ids:', ids)
    link_idx = {'base': 0, 'switch': 0, 'S': 1, 'L': 2, 'U': 3, 'R': 4, 'B': 5, 'T': 6}
    files = {'base': 'MPX3500_base.stl', 'S': 'MPX3500_S.stl', 'L': 'MPX3500_L.stl',
             'U': 'MPX3500_U.stl', 'R': 'MPX3500_R.stl', 'B': 'MPX3500_B.stl', 'T': 'MPX3500_T.stl'}
    parts = {}
    for name, k in ids.items():
        T = tris[tl == k]
        W = to_world(T.reshape(-1, 3))
        inv = np.linalg.inv(frames[link_idx[name]])
        Lc = (inv[:3, :3] @ W.T).T + inv[:3, 3]
        parts.setdefault('base' if name == 'switch' else name, []).append(Lc.reshape(-1, 3, 3))
    for name, fname in files.items():
        tr = np.vstack(parts[name])
        before = len(tr)
        cell = TARGET_CELL if before > 25000 else TARGET_CELL / 2
        tr = decimate(tr, cell)
        save_stl(os.path.join(out, fname), tr)
        print(f'{name:5s} {before:7d} -> {len(tr):6d} triangles')


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else '.'
    main(src, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets'))
