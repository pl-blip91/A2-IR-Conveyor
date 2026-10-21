"""Teach pendant for the loading station.

One window to run the job, jog the arm, send it to a joint state, press the simulated
e-stop and exercise the safety sensors.
"""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk
from typing import Optional

import numpy as np

from workcell import loading_config as lc
from workcell.loading_station import LoadingStation, RunState

# state -> (background, foreground) of the state banner
STATE_COLOURS = {
    "IDLE": ("#5b6068", "white"),
    "RUNNING": ("#1e8e3e", "white"),
    "BLOCKED": ("#e8a100", "black"),
    "PROTECTIVE": ("#e8710a", "white"),
    "STOPPED": ("#e8a100", "black"),
    "ESTOPPED": ("#c5221f", "white"),
    "FAULT": ("#c5221f", "white"),
    "DONE": ("#1a73e8", "white"),
}
STATE_HELP = {
    "IDLE": "Ready. Press Start, or jog the arm.",
    "RUNNING": "Job running.",
    "BLOCKED": "Holding still: an obstacle is in the planned path. It continues when the path is clear.",
    "PROTECTIVE": "Safety stop (light curtain / person in zone). Clear the zone, then Reset, then Resume.",
    "STOPPED": "Reset done. Press Resume to continue from where it stopped.",
    "ESTOPPED": "EMERGENCY STOP. Release the e-stop, then Reset, then Resume.",
    "FAULT": "Job aborted by a fault. Press Re-arm to recover.",
    "DONE": "Job complete. Jog the arm, or Re-arm to run again.",
}
AXES = ("X", "Y", "Z")
REFRESH_EVERY = 2  # redraw the readouts every N ticks


class TeachPendant:
    """The teach/jog window. station is a :class:LoadingStation."""

    def __init__(self, root: tk.Tk, station: LoadingStation):
        self.root = root
        self.station = station
        self._running = True
        self._ticks = 0
        self._log_shown = 0
        root.title("MH5 loading station - teach pendant")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<FocusOut>", lambda _e: station.stop_jog())

        self.jog_buttons: list[tk.Button] = []
        self._build()
        self.refresh()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        r = self.root
        r.columnconfigure(0, weight=0)
        r.columnconfigure(1, weight=1)
        r.rowconfigure(1, weight=1)

        self.banner = tk.Label(r, text="IDLE", font=("Helvetica", 22, "bold"), pady=6)
        self.banner.grid(row=0, column=0, columnspan=2, sticky="ew")

        left = ttk.Frame(r, padding=6)
        left.grid(row=1, column=0, sticky="ns")
        right = ttk.Frame(r, padding=6)
        right.grid(row=1, column=1, sticky="nsew")
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_run_controls(left)
        self._build_joint_jog(left)
        self._build_cart_jog(left)
        self._build_status(right)
        self._build_goto(right)
        self._build_safety_sim(right)
        self._build_log(right)

    def _build_run_controls(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="Job and stop controls", padding=6)
        box.grid(sticky="ew", pady=3)
        self.btn_start = ttk.Button(box, text="Start", command=self._start)
        self.btn_rearm = ttk.Button(box, text="Re-arm", command=self._rearm)
        self.btn_start.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        self.btn_rearm.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        self.btn_estop = tk.Button(box, text="E-STOP", bg="#c5221f", fg="white", activebackground="#8c1814",
                                   activeforeground="white", font=("Helvetica", 16, "bold"), height=1,
                                   command=self.station.emergency_stop)
        self.btn_estop.grid(row=1, column=0, columnspan=2, padx=2, pady=4, sticky="ew")
        self.btn_release = ttk.Button(box, text="1. Release e-stop", command=self.station.release_estop)
        self.btn_reset = ttk.Button(box, text="2. Reset (acknowledge)", command=self._reset)
        self.btn_resume = ttk.Button(box, text="3. Resume", command=self._resume)
        for i, b in enumerate((self.btn_release, self.btn_reset, self.btn_resume)):
            b.grid(row=2 + i, column=0, columnspan=2, padx=2, pady=1, sticky="ew")
        box.columnconfigure((0, 1), weight=1)

    def _hold_button(self, parent, text: str, press, width: int = 4) -> tk.Button:
        b = tk.Button(parent, text=text, width=width)
        b.bind("<ButtonPress-1>", lambda _e: press())
        b.bind("<ButtonRelease-1>", lambda _e: self.station.stop_jog())
        b.bind("<Leave>", lambda _e: self.station.stop_jog())
        self.jog_buttons.append(b)
        return b

    def _build_joint_jog(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="Joint jog (hold to move)", padding=6)
        box.grid(sticky="ew", pady=3)
        lo, hi = self.station.arm.qlim_array
        self.joint_vars, self.joint_bars = [], []
        for j in range(6):
            ttk.Label(box, text=f"J{j + 1}").grid(row=j, column=0, padx=2)
            self._hold_button(box, "-", lambda j=j: self._jog("joint", j, -1)).grid(row=j, column=1, padx=1)
            self._hold_button(box, "+", lambda j=j: self._jog("joint", j, +1)).grid(row=j, column=2, padx=1)
            var = tk.StringVar(value="0.0")
            ttk.Label(box, textvariable=var, width=8, anchor="e").grid(row=j, column=3, padx=2)
            bar = ttk.Progressbar(box, length=90, maximum=100)
            bar.grid(row=j, column=4, padx=2)
            self.joint_vars.append(var)
            self.joint_bars.append((bar, lo[j], hi[j]))

    def _build_cart_jog(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="Cartesian jog (world frame, hold to move)", padding=6)
        box.grid(sticky="ew", pady=3)
        for a, name in enumerate(AXES):
            ttk.Label(box, text=name).grid(row=0, column=2 * a, columnspan=2)
            self._hold_button(box, "-", lambda a=a: self._jog("cart", a, -1), 3).grid(row=1, column=2 * a, padx=1)
            self._hold_button(box, "+", lambda a=a: self._jog("cart", a, +1), 3).grid(row=1, column=2 * a + 1, padx=1)
        self.tcp_var = tk.StringVar(value="")
        ttk.Label(box, textvariable=self.tcp_var).grid(row=2, column=0, columnspan=6, pady=(4, 0))

    def _build_goto(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="Go to joint state (degrees)", padding=6)
        box.grid(row=1, column=0, sticky="ew", pady=3)
        self.goto_entries = []
        for j in range(6):
            ttk.Label(box, text=f"J{j + 1}").grid(row=0, column=j)
            e = ttk.Entry(box, width=6, justify="right")
            e.insert(0, f"{np.degrees(lc.Q_TRAVEL[j]):.0f}")
            e.grid(row=1, column=j, padx=1)
            self.goto_entries.append(e)
        self.btn_go = ttk.Button(box, text="Go", command=self._goto)
        self.btn_home = ttk.Button(box, text="Home", command=self.station.go_home)
        self.btn_copy = ttk.Button(box, text="Copy current", command=self._copy_current)
        self.btn_go.grid(row=1, column=6, padx=(8, 2), sticky="ew")
        self.btn_home.grid(row=1, column=7, padx=2, sticky="ew")
        self.btn_copy.grid(row=1, column=8, padx=2, sticky="ew")

    def _build_status(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="State", padding=6)
        box.grid(row=0, column=0, sticky="ew", pady=3)
        self.status_vars = {}
        fields = (("Phase", "phase"), ("Cars placed", "cars"), ("Base x, y, yaw", "base"),
                  ("Tool x, y, z", "tcp"), ("Manipulability", "manip"), ("Vision error", "vision"),
                  ("Re-plans", "replans"), ("E-stop input", "estop"), ("Light curtain", "curtain"),
                  ("Unsafe zone", "zone"), ("Obstacle", "obstacle"), ("Blocked because", "block"),
                  ("Fault", "fault"), ("Last event", "message"))
        for i, (label, key) in enumerate(fields):
            ttk.Label(box, text=label + ":").grid(row=i // 2, column=(i % 2) * 2, sticky="w", padx=(0, 4))
            var = tk.StringVar(value="-")
            lab = ttk.Label(box, textvariable=var, width=34 if i % 2 == 0 else 38, anchor="w")
            lab.grid(row=i // 2, column=(i % 2) * 2 + 1, sticky="w", padx=(0, 10))
            self.status_vars[key] = var
        self.help_var = tk.StringVar(value="")
        ttk.Label(box, textvariable=self.help_var, wraplength=640, foreground="#444").grid(
            row=10, column=0, columnspan=4, sticky="w", pady=(6, 0))

    def _build_safety_sim(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="Safety simulation (inputs for the demonstration)", padding=6)
        box.grid(row=4, column=0, sticky="ew", pady=3)
        for c in range(4):
            box.columnconfigure(c, weight=1)
        ttk.Button(box, text="Person walks into cell", command=self.station.person_walk_in).grid(
            row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(box, text="Person leaves", command=self.station.person_walk_out).grid(
            row=0, column=1, sticky="ew", padx=2, pady=2)
        self.curtain_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(box, text="Force light curtain broken", variable=self.curtain_var,
                        command=lambda: self.station.trip_light_curtain(self.curtain_var.get())).grid(
            row=0, column=2, columnspan=2, sticky="w", padx=2)
        ttk.Button(box, text="Drop obstacle in path", command=self._drop_obstacle).grid(
            row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(box, text="Remove obstacle", command=self.station.remove_obstacle).grid(
            row=1, column=1, sticky="ew", padx=2, pady=2)
        ttk.Label(box, text="Speed").grid(row=1, column=2, sticky="e")
        self.speed = tk.DoubleVar(value=self.station.speed_scale)
        ttk.Scale(box, from_=0.25, to=3.0, variable=self.speed, command=self._set_speed).grid(
            row=1, column=3, sticky="ew", padx=4)

    def _build_log(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="Event log", padding=4)
        box.grid(row=3, column=0, sticky="nsew", pady=3)
        box.rowconfigure(0, weight=1)
        box.columnconfigure(0, weight=1)
        self.log = tk.Text(box, height=7, width=80, state="disabled", font=("Courier", 9))
        sb = ttk.Scrollbar(box, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

    # ---------------------------------------------------------------- commands
    def _start(self) -> None:
        try:
            self.station.start()
        except RuntimeError as err:
            self.station._log(f"start refused: {err}")

    def _rearm(self) -> None:
        if not self.station.rearm():
            self.station._log("re-arm refused (job running, stop active or person in zone)")

    def _reset(self) -> None:
        if not self.station.reset():
            self.station._log("reset refused: release the e-stop and clear the zone first")

    def _resume(self) -> None:
        if not self.station.resume():
            self.station._log("resume refused: reset first")

    def _jog(self, kind: str, index: int, sign: int) -> None:
        self.station.set_jog(kind, index, sign)

    def _goto(self) -> None:
        try:
            q = np.radians([float(e.get()) for e in self.goto_entries])
        except ValueError:
            self.station._log("go to: enter six numbers (degrees)")
            return
        self.station.goto_q(q)

    def _copy_current(self) -> None:
        for e, qd in zip(self.goto_entries, np.degrees(self.station.arm.q)):
            e.delete(0, "end")
            e.insert(0, f"{qd:.1f}")

    def _drop_obstacle(self) -> None:
        if self.station.drop_obstacle_in_path() is None:
            self.station._log("nothing is moving: start the job or send the arm somewhere first")

    def _set_speed(self, _value=None) -> None:
        self.station.speed_scale = round(float(self.speed.get()), 2)

    # ----------------------------------------------------------------- display
    def refresh(self) -> None:
        st = self.station
        s = st.status()
        bg, fg = STATE_COLOURS.get(s["state"], ("#5b6068", "white"))
        self.banner.configure(text=s["state"] + (f"  -  {st.phase}" if s["state"] in ("ESTOPPED", "PROTECTIVE") else ""),
                              bg=bg, fg=fg)
        self.help_var.set(STATE_HELP.get(s["state"], ""))
        v = self.status_vars
        v["phase"].set(s["phase"])
        v["cars"].set(f"{s['cars_placed']} / {st.n_cars}" + ("  (holding a car)" if s["holding_car"] else ""))
        bx, by, byaw = s["base_xy_yaw"]
        v["base"].set(f"{bx:.2f}, {by:.2f}, {np.degrees(byaw):.0f} deg")
        v["tcp"].set("{:.3f}, {:.3f}, {:.3f} m".format(*s["tcp_xyz"]))
        m = s["manipulability"]
        v["manip"].set("-" if m != m else f"{m:.4f}")
        e = s["vision_error_px"]
        v["vision"].set("-" if e != e else f"{e:.2f} px")
        v["replans"].set(str(s["replans"]))
        v["estop"].set("PRESSED" if s["estop_pressed"] else "released")
        v["curtain"].set("BROKEN" if s["curtain_broken"] else "clear")
        v["zone"].set("OCCUPIED" if s["zone_occupied"] else "clear")
        v["obstacle"].set("in the cell" if s["obstacle_active"] else "none")
        v["block"].set(s["block_reason"] or "-")
        v["fault"].set(s["fault"] or "-")
        v["message"].set(s["message"] or "-")
        self.tcp_var.set("tool: {:.3f}, {:.3f}, {:.3f} m".format(*s["tcp_xyz"]))
        for var, (bar, lo, hi), q in zip(self.joint_vars, self.joint_bars, st.arm.q):
            var.set(f"{np.degrees(q):7.1f}")
            bar["value"] = 100 * (q - lo) / (hi - lo)

        state = st.run_state
        manual = st.manual_allowed
        idle = state is RunState.IDLE and st.inputs_clear
        self._enable(self.btn_start, idle)
        self._enable(self.btn_rearm, state in (RunState.IDLE, RunState.DONE, RunState.FAULT) and st.inputs_clear)
        self._enable(self.btn_release, s["estop_pressed"])
        self._enable(self.btn_reset, state in (RunState.ESTOPPED, RunState.PROTECTIVE) and st.inputs_clear)
        self._enable(self.btn_resume, state is RunState.STOPPED and st.inputs_clear)
        for w in (self.btn_go, self.btn_home):
            self._enable(w, manual)
        for b in self.jog_buttons:
            b.configure(state="normal" if manual else "disabled")

        events = list(st.events)
        if len(events) != self._log_shown or (events and events[-1] != getattr(self, "_last_event", None)):
            self.log.configure(state="normal")
            self.log.delete("1.0", "end")
            self.log.insert("end", "\n".join(events[-40:]))
            self.log.see("end")
            self.log.configure(state="disabled")
            self._log_shown = len(events)
            self._last_event = events[-1] if events else None

    @staticmethod
    def _enable(widget, on: bool) -> None:
        widget.state(["!disabled"] if on else ["disabled"])

    # -------------------------------------------------------------------- loop
    def start_loop(self) -> None:
        self._schedule(1)

    def _schedule(self, delay_ms: int) -> None:
        if self._running:
            self.root.after(delay_ms, self._step)

    def _step(self) -> None:
        if not self._running:
            return
        t0 = time.perf_counter()
        self.station.tick()
        self._ticks += 1
        if self._ticks % REFRESH_EVERY == 0:
            self.refresh()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._schedule(max(1, int(self.station.dt * 1000 - elapsed_ms)))

    def close(self) -> None:
        self._running = False
        self.station.stop_jog()
        self.root.destroy()


def run_gui(station: LoadingStation, root: Optional[tk.Tk] = None) -> None:
    """Open the teach pendant for ``station`` and run until the window is closed."""
    root = root or tk.Tk()
    app = TeachPendant(root, station)
    root.update()
    root.lift()
    root.attributes("-topmost", True)
    root.after(800, lambda: root.attributes("-topmost", False))
    app.start_loop()
    root.mainloop()
