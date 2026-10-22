import numpy as np
from spatialgeometry import Cuboid, Cylinder, Sphere
from spatialmath import SE3
from spatialmath.base import transl

from shared import line_config as cfg
from shared.scene import PARK, Assembly, box

try:  # optional: physical e-stop on a gamepad button
    import pygame
except Exception:  # noqa: BLE001
    pygame = None

READY, RUNNING, STOPPED = "READY", "RUNNING", "STOPPED"
GREEN, RED, AMBER = (0.1, 0.9, 0.2, 1.0), (1.0, 0.05, 0.05, 1.0), (1.0, 0.65, 0.0, 1.0)
FENCE_COLOUR = (1.0, 0.45, 0.0, 1.0)
GATE_COLOUR = (1.0, 0.85, 0.0, 1.0)
MESH_COLOUR = (0.25, 0.25, 0.27, 0.35)  # see-through wire-mesh infill of the fence panels
FENCE_POST_GAP = 1.5
FENCE_H = 1.4  # guard fence height (m)
FENCE_Y = cfg.FENCE_Y
CURTAIN_INSET = 0.15  # the light-curtain plane stands this far inside the door line (m)
CURTAIN_ABS_Y = FENCE_Y - CURTAIN_INSET

HW_ESTOP_BUTTON = 0  # gamepad button index used as the physical e-stop input (Lab 11)
HW_ESTOP_INVERT = False  # True if the input reads 1 when the e-stop is healthy

PERSON_R = 0.18  # body radius used for the beam-break test
PERSON_START = FENCE_Y + 1.6  # |y| of the person while outside the cell
PERSON_INSIDE = 1.0  # |y| of the person once inside the cell
PERSON_SPEED = 0.6  # m/s
WARN_ZONE = 1.0  # strip outside the fence (m) in which the cell is slowed
SLOW_SCALE = 0.5  # fraction of normal speed inside the warning strip
BEAM_Z = (0.25, 0.55, 0.85)


# =============================================================================
# State machine
# =============================================================================
class SafetySystem:
    """E-stop latch state machine. No loops: update() is called once per frame and only changes state on events.

        READY --START--> RUNNING --(e-stop / light curtain / area scanner)--> STOPPED --RESET (cause cleared)--> READY

    A stop is LATCHED: releasing the e-stop or clearing the zone never restarts the cell. RESET is refused while any cause is
    still active, and RESET only goes to READY (automatic motion still inhibited); a separate START is needed to run."""

    def __init__(self):
        self.state, self.latched, self.active = READY, [], []
        self.gui_estop = self.hw_pressed = self.zone_tripped = False
        self.reset_req = self.start_req = False
        self.message = "READY: press Start production"

    # --- requests (called from the GUI; they only set flags, the state changes in update()) ---
    def press_gui_estop(self):
        self.gui_estop = True

    def release_gui_estop(self):
        self.gui_estop = False

    def request_reset(self):
        self.reset_req = True

    def request_start(self):
        self.start_req = True

    def update(self, hw_pressed, zone_causes):
        """Returns True on the frame the system enters STOPPED."""
        self.hw_pressed, self.zone_tripped = hw_pressed, bool(zone_causes)
        active = []
        if self.gui_estop:
            active.append("GUI E-STOP")
        if hw_pressed:
            active.append("HW E-STOP")
        active.extend(zone_causes)  # light curtains / area scanner: one cause each
        self.active = active
        for cause in active:
            if cause not in self.latched:
                self.latched.append(cause)
                self.message = f"STOP: {cause} - motion halted and latched"
        newly_stopped = bool(active) and self.state != STOPPED
        if newly_stopped:
            self.state = STOPPED
        reset, start = self.reset_req, self.start_req
        self.reset_req = self.start_req = False
        if self.state == STOPPED:
            if reset:
                if active:
                    self.message = "RESET refused - still active: " + ", ".join(active)
                else:
                    self.latched.clear()
                    self.state = READY
                    self.message = "Reset acknowledged - system READY. Press Resume to carry on."
            elif start:
                self.message = "START refused - release the cause, press RESET first"
            elif not active and not newly_stopped:
                self.message = "Cause cleared - press RESET to acknowledge (the cell does NOT restart by itself)"
        elif self.state == READY and start:
            if active:
                self.message = "START refused - still active: " + ", ".join(active)
            else:
                self.state = RUNNING
                self.message = "RUNNING"
        return newly_stopped

    @property
    def estopped(self):
        return any("E-STOP" in c for c in self.latched)


class HardwareEstop:
    """NON-BLOCKING reader of the physical e-stop input (a gamepad button read with pygame, Lab 11).
    pressed() is called once per frame and never waits. If the gamepad is unplugged while running, the e-stop counts as
    PRESSED (fail-safe). With no gamepad connected only the simulated GUI e-stop is available.

    macOS: SDL (inside pygame) and Tk both want to own the Cocoa application object, and whichever starts second must
    find the first one's.  ProductionLine.py therefore creates the teach pendant's Tk window BEFORE this class starts
    SDL; only SDL's joystick part is started, plus its display (needed to pump joystick events) only if a gamepad is
    plugged in, and never audio.
    enabled=False (ProductionLine.py --no-gamepad) skips pygame altogether."""

    def __init__(self, button=HW_ESTOP_BUTTON, invert=HW_ESTOP_INVERT, enabled=True):
        self.button, self.invert = button, invert
        self.joy, self.status, self.connected = None, "NOT CONNECTED", False
        if not enabled:
            self.status = "DISABLED (--no-gamepad)"
            print("Hardware e-stop disabled (--no-gamepad) - only the simulated GUI e-stop is active.")
            return
        if pygame is None:
            print("WARNING: pygame is not installed - only the simulated GUI e-stop is active.")
            return
        try:
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                pygame.display.init()  # SDL video is only started when there IS a gamepad: event.pump() needs it
                self.joy = pygame.joystick.Joystick(0)
                self.joy.init()
                self.connected, self.status = True, "released"
                print(f"Hardware e-stop: gamepad '{self.joy.get_name()}', button {button}")
            else:
                print("WARNING: no gamepad found - only the simulated GUI e-stop is active.")
        except Exception as err:  # noqa: BLE001
            print(f"WARNING: gamepad unavailable ({err}) - only the simulated GUI e-stop is active.")

    def pressed(self):
        """True = e-stop pressed (or input lost). Never blocks."""
        if not self.connected:
            return False
        try:
            pygame.event.pump()
            state = bool(self.joy.get_button(self.button)) != self.invert
        except Exception:  # noqa: BLE001
            self.status = "LINK LOST"
            return True
        self.status = "PRESSED" if state else "released"
        return state

    def close(self):
        if pygame is not None:
            try:
                pygame.quit()
            except Exception:  # noqa: BLE001
                pass


# =============================================================================
# Person, light curtains, stack lights, e-stop posts
# =============================================================================
class Person(Assembly):
    """Simulated person who can walk in through EITHER door (loading gate or sorting door).  The door can only be
    changed while the person is outside."""

    def __init__(self, env):
        super().__init__(env, [
            (Cylinder(radius=0.11, length=0.80, color=[0.10, 0.15, 0.45, 1.0]), SE3(0, 0, 0.40)),  # legs
            (Cylinder(radius=0.17, length=0.50, color=[0.95, 0.80, 0.05, 1.0]), SE3(0, 0, 1.05)),  # hi-vis torso
            (Sphere(radius=0.11, color=[0.90, 0.70, 0.55, 1.0]), SE3(0, 0, 1.42)),  # head
            (Cylinder(radius=0.12, length=0.06, color=[1.0, 1.0, 1.0, 1.0]), SE3(0, 0, 1.52))])  # hard hat
        self.door = "sort"
        self.abs_y, self.target = PERSON_START, PERSON_START
        self._place()

    def _place(self):
        d = cfg.DOORS[self.door]
        self.T = SE3(d["x"], d["side"] * self.abs_y, 0).A

    @property
    def outside(self):
        return self.abs_y >= PERSON_START - 1e-9 and self.target >= PERSON_START - 1e-9

    def choose_door(self, key):
        if key == self.door:
            return True
        if not self.outside:
            return False
        self.door = key
        self._place()
        return True

    def cycle_door(self):
        keys = list(cfg.DOORS)
        return self.choose_door(keys[(keys.index(self.door) + 1) % len(keys)])

    def go_inside(self, door=None):
        if door is not None and not self.choose_door(door):
            return False
        self.target = PERSON_INSIDE
        return True

    def go_outside(self):
        self.target = PERSON_START

    def toggle(self):
        self.go_inside() if self.target >= PERSON_START else self.go_outside()

    def update(self, dt):
        if self.abs_y == self.target:
            return
        step = PERSON_SPEED * dt
        self.abs_y = float(self.abs_y + np.clip(self.target - self.abs_y, -step, step))
        self._place()

    @property
    def xy(self):
        d = cfg.DOORS[self.door]
        return d["x"], d["side"] * self.abs_y

    def in_curtain(self):
        """True while any part of the person is at / beyond the light-curtain plane (beam broken or inside the cell)."""
        return self.abs_y - PERSON_R <= CURTAIN_ABS_Y

    def in_load_zone(self):
        """True while the person's footprint overlaps the loading zone watched by the area scanners (Lachlan)."""
        px, py = self.xy
        cx = min(max(px, cfg.LOAD_ZONE_X[0]), cfg.LOAD_ZONE_X[1])
        cy = min(max(py, cfg.LOAD_ZONE_Y[0]), cfg.LOAD_ZONE_Y[1])
        return bool(np.hypot(px - cx, py - cy) < PERSON_R)

    def in_warning_zone(self):
        return self.abs_y - PERSON_R < FENCE_Y + WARN_ZONE

    @property
    def status(self):
        name = cfg.DOORS[self.door]["name"]
        if abs(self.abs_y - self.target) > 1e-9:
            return f"{'entering' if self.target < self.abs_y else 'leaving'} via the {name}"
        return f"outside (at the {name})" if self.abs_y >= PERSON_START - 1e-9 else f"INSIDE the cell ({name})"


class LightCurtain:
    """Emitter / receiver posts with three beams across the door (Lab 3 Cuboids). Green beams = clear, red = tripped."""

    def __init__(self, env, x0, x1, y):
        self.green, self.red, self.tripped = [], [], None
        for z in BEAM_Z:
            for store, colour in ((self.green, GREEN), (self.red, RED)):
                beam = Cuboid(scale=[x1 - x0, 0.012, 0.012], color=list(colour))
                env.add(beam)
                store.append((beam, transl((x0 + x1) / 2, y, z)))
        for x in (x0, x1):
            box(env, (0.07, 0.07, 1.0), (x, y, 0.5), (0.10, 0.10, 0.12, 1.0))
            box(env, (0.075, 0.075, 0.06), (x, y, 1.03), RED)
        self.set(False)

    def set(self, tripped):
        if tripped == self.tripped:
            return
        self.tripped = tripped
        for beam, T in self.green:
            beam.T = PARK if tripped else T
        for beam, T in self.red:
            beam.T = T if tripped else PARK


class StackLight:
    """Red / amber / green tower: red = e-stop or fault, amber = ready / protective stop / holding, green = running."""

    def __init__(self, env, x, y):
        box(env, (0.06, 0.06, 1.2), (x, y, 0.6), (0.35, 0.35, 0.38, 1.0))
        self.lit, self.state = {}, None
        for name, z, colour in ((STOPPED, 1.46, RED), (READY, 1.34, AMBER), (RUNNING, 1.22, GREEN)):
            dim = [0.3 * c for c in colour[:3]] + [1.0]
            env.add(Cylinder(radius=0.05, length=0.10, color=dim, pose=SE3(x, y, z)))
            lamp = Cylinder(radius=0.055, length=0.105, color=list(colour))
            env.add(lamp)
            self.lit[name] = (lamp, transl(x, y, z))
        self.set(READY)

    def set(self, state):
        if state == self.state:
            return
        self.state = state
        for name, (lamp, T) in self.lit.items():
            lamp.T = T if name == state else PARK


class EStopPost:
    """Red mushroom e-stop on a yellow box on a grey post.  The button sinks when the line's e-stop is pressed."""

    def __init__(self, env, x, y):
        box(env, (0.06, 0.06, 0.8), (x, y, 0.4), (0.4, 0.4, 0.42, 1.0))
        box(env, (0.14, 0.14, 0.12), (x, y, 0.86), GATE_COLOUR)
        self.up = (Cylinder(radius=0.05, length=0.04, color=[0.85, 0.05, 0.05, 1.0]), transl(x, y, 0.94))
        self.down = (Cylinder(radius=0.05, length=0.02, color=[0.45, 0.02, 0.02, 1.0]), transl(x, y, 0.925))
        env.add(self.up[0])
        env.add(self.down[0])
        self.pressed = None
        self.set(False)

    def set(self, pressed):
        if pressed == self.pressed:
            return
        self.pressed = pressed
        self.up[0].T = PARK if pressed else self.up[1]
        self.down[0].T = self.down[1] if pressed else PARK


# =============================================================================
# Guard fence
# =============================================================================
def fence_run(sb, a, b, height=FENCE_H):
    """Straight run of orange guard fence from a=(x, y) to b=(x, y) in any direction: posts every FENCE_POST_GAP, a top,
    middle and bottom rail and a see-through mesh infill."""
    (xa, ya), (xb, yb) = a, b
    length = float(np.hypot(xb - xa, yb - ya))
    yaw = float(np.arctan2(yb - ya, xb - xa))
    mid = ((xa + xb) / 2, (ya + yb) / 2)
    for z in (0.12, height / 2, height - 0.03):
        sb.box((length, 0.04, 0.05), (mid[0], mid[1], z), FENCE_COLOUR, yaw)
    sb.box((length, 0.01, height - 0.17), (mid[0], mid[1], 0.12 + (height - 0.17) / 2), MESH_COLOUR, yaw)
    n = max(1, int(np.ceil(length / FENCE_POST_GAP)))
    for k in range(n + 1):
        px, py = xa + (xb - xa) * k / n, ya + (yb - ya) * k / n
        sb.box((0.06, 0.06, height), (px, py, height / 2), FENCE_COLOUR)
        sb.box((0.16, 0.16, 0.016), (px, py, 0.008), FENCE_COLOUR)  # foot plate


def sign(sb, x, y, yaw=0.0, z=1.0, colour=(0.98, 0.85, 0.05, 1.0)):
    """Yellow warning sign with a black symbol, hung on the fence."""
    sb.box((0.34, 0.012, 0.26), (x, y, z), colour, yaw)
    sb.box((0.06, 0.014, 0.12), (x, y, z), (0.1, 0.1, 0.1, 1.0), yaw)


def build_loader_zone_scenery(sb):
    """Floor-level laser area scanners at the belt-side corners of the loading zone, warning signs and hazard strips.
    The belt side of the loading zone cannot be fenced because the line carries on past it, so two scanners watch the open
    edge; they are what the 'loading zone occupied' safety input models (Lachlan)."""
    yellow, black = (0.95, 0.80, 0.10, 1.0), (0.1, 0.1, 0.1, 1.0)
    x0, x1 = cfg.LOAD_ZONE_X
    y0, y1 = cfg.LOAD_ZONE_Y
    for sx, side in ((x0 + 0.1, 1), (x1 - 0.1, -1)):
        sb.box((0.10, 0.10, 0.18), (sx, y0 + 0.08, 0.09), yellow)
        sb.box((0.11, 0.11, 0.025), (sx, y0 + 0.08, 0.19), black)
        sb.box((0.02, 0.06, 0.05), (sx - side * 0.055, y0 + 0.08, 0.12), (1.0, 0.1, 0.1, 1.0))
    w, z = 0.06, 0.009  # hazard strips stand just proud of the painted floor (shared/factory.py)
    for size, centre in (((x1 - x0, w, 0.006), ((x0 + x1) / 2, y1 - w, z)),
                         ((x1 - x0, w, 0.006), ((x0 + x1) / 2, y0 + w / 2, z)),
                         ((w, y1 - y0, 0.006), (x0 + w, (y0 + y1) / 2, z)),
                         ((w, y1 - y0, 0.006), (x1 - w, (y0 + y1) / 2, z))):
        sb.box(size, centre, yellow)


def build_safety_cell(env, sb, stamp_x, sort_x):
    """Static parts go into the StaticScene `sb`, the e-stop posts (their buttons move) into env.
    Orange guard fence round the whole line, with the octagonal enclosure round the MPX3500 sized to its full reach,
    a yellow gate at the loader (+y) and a yellow door at the UR3e (-y), an e-stop post at every station.

    Returns (e-stop posts, (sorting door x0, x1), (loading gate x0, x1))."""
    x0, x1 = cfg.FENCE_X0, cfg.FENCE_X1
    door_l, door_r = sort_x - cfg.DOOR_W / 2, sort_x + cfg.DOOR_W / 2
    g0, g1 = cfg.GATE_X
    guard = cfg.PAINT_GUARD
    Y = FENCE_Y
    # -y side: fence, gap for the sorting door, fence
    fence_run(sb, (x0, -Y), (door_l, -Y))
    fence_run(sb, (door_r, -Y), (x1, -Y))
    # +y side: fence, gap for the loading gate, fence, then out round the MPX3500 (octagon) and back
    fence_run(sb, (x0, Y), (g0, Y))
    fence_run(sb, (g1, Y), guard[0])
    for a, b in zip(guard[:-1], guard[1:]):
        fence_run(sb, a, b)
    fence_run(sb, guard[-1], (x1, Y))
    fence_run(sb, (x0, -Y), (x0, Y))
    fence_run(sb, (x1, -Y), (x1, Y))
    for gx0, gx1, gy in ((door_l, door_r, -Y), (g0, g1, Y)):  # the two yellow gates: two posts, a top bar, a middle bar
        for gx in (gx0, gx1):
            sb.box((0.06, 0.06, FENCE_H), (gx, gy, FENCE_H / 2), GATE_COLOUR)
        sb.box((gx1 - gx0, 0.04, 0.06), ((gx0 + gx1) / 2, gy, FENCE_H), GATE_COLOUR)
        sb.box((gx1 - gx0, 0.03, 0.04), ((gx0 + gx1) / 2, gy, 0.55), GATE_COLOUR)
    # warning signs on the outside of the fence
    sign(sb, (x0 + g0) / 2, Y + 0.03)
    sign(sb, (g1 + guard[0][0]) / 2, Y + 0.03)
    sign(sb, door_l - 0.6, -Y - 0.03)
    for (ax, ay), (bx, by) in zip(guard[1:-2], guard[2:-1]):  # one on each outer face of the paint enclosure
        sign(sb, (ax + bx) / 2, (ay + by) / 2 + 0.03, float(np.arctan2(by - ay, bx - ax)))
    build_loader_zone_scenery(sb)
    estops = [EStopPost(env, g0 - 0.30, Y + 0.80),  # loader (beside the loading gate)
              EStopPost(env, cfg.PAINT_X, cfg.PAINT_BULGE_Y + 0.6),  # MPX3500 (outside its enclosure)
              EStopPost(env, stamp_x, -(Y + 0.6)),  # FANUC (aisle side: the paint enclosure is behind it on +y)
              EStopPost(env, door_l - 0.8, cfg.SORT_SIDE * (Y + 0.6))]  # UR3e (beside the door)
    return estops, (door_l, door_r), (g0, g1)
