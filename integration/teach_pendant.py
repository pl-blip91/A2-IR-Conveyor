from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk

import numpy as np

from integration.production_cell import DT, JOG_CART_SPEED, JOG_JOINT_SPEED, ToyCarCell

# ---------------------------------------------------------------------------------------------------------------- look
BG = "#f3f5f8"  # window background
PANEL = "#ffffff"  # page / card background
TEXT = "#1f2933"
MUTED = "#6b7785"
LINE = "#d5dbe3"
ACCENT = "#1a73e8"
GREEN, AMBER, RED, GREY = "#1e8e3e", "#e8a100", "#c5221f", "#9aa5b1"
FONT = "Helvetica"

# state -> (background, foreground) of the state banner
STATE_COLOURS = {
    "IDLE": ("#5b6068", "white"),
    "MANUAL": ("#5b6068", "white"),
    "RUNNING": (GREEN, "white"),
    "BLOCKED": (AMBER, "black"),
    "PROTECTIVE": ("#e8710a", "white"),
    "STOPPED": (AMBER, "black"),
    "ESTOPPED": (RED, "white"),
    "FAULT": (RED, "white"),
    "DONE": (ACCENT, "white"),
}
STATE_HELP = {
    "IDLE": "Ready. Press Start, or open the Jog page and hold a button to move a robot.",
    "MANUAL": "Line enabled, no production running. Jog or move any robot from the Jog / Move pages.",
    "RUNNING": "Production running. Jogging is locked out until it completes.",
    "BLOCKED": "A robot is holding still: an obstacle is in its path. It carries on when the path is clear.",
    "PROTECTIVE": "Safety stop (light curtain / area scanner). Clear the zone, then 2 Reset, then 3 Resume.",
    "STOPPED": "Reset done. Press 3 Resume to carry on from exactly where the line stopped.",
    "ESTOPPED": "EMERGENCY STOP: everything is frozen. 1 Release the e-stop, 2 Reset, 3 Resume.",
    "FAULT": "Production aborted by a fault (see Log). Manual control is available; Re-arm to run again.",
    "DONE": "Production complete. Jog the robots, or Re-arm to put the cars back and run again.",
}
AXES = ("X", "Y", "Z")
SIM_SPEEDS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)  # simulation speed slider steps
JOG_PAGE_FIELDS = ("Tool x, y, z",)  # robot_fields() shown on the Jog page instead of the Move page
SAFETY_FIELDS = ("E-stop", "Light curtains", "Area scanner", "Person")  # line_fields() shown on the Safety page
ESTOP_RED, ESTOP_RED_DOWN, ESTOP_YELLOW = "#d50000", "#8b0000", "#ffd600"
REFRESH_EVERY = 2  # redraw the readouts every N ticks
KEY_OBSTACLES = {"u": ("mh5", True), "i": ("mh5", False), "j": ("mpx", True), "h": ("mpx", False),
                 "o": ("fanuc", True), "p": ("fanuc", False), "k": ("ur", True), "l": ("ur", False)}
WRAP = 330  # wrap length (px) of long values


def apply_style(root):
    """One light theme for every widget.  'clam' is drawn by Tk itself, so colours look the same on macOS, Windows and
    Linux (the native macOS theme ignores button colours)."""
    st = ttk.Style(root)
    st.theme_use("clam")
    root.configure(bg=BG)
    st.configure(".", background=BG, foreground=TEXT, font=(FONT, 10), bordercolor=LINE, lightcolor=BG, darkcolor=BG)
    st.configure("TFrame", background=BG)
    st.configure("Panel.TFrame", background=PANEL)
    st.configure("TLabel", background=BG)
    st.configure("Panel.TLabel", background=PANEL)
    st.configure("Key.TLabel", background=PANEL, foreground=MUTED)
    st.configure("Muted.TLabel", background=BG, foreground=MUTED, font=(FONT, 9))
    st.configure("PanelMuted.TLabel", background=PANEL, foreground=MUTED, font=(FONT, 9))
    st.configure("Value.TLabel", background=PANEL, font=(FONT, 10, "bold"))
    st.configure("Head.TLabel", background=PANEL, foreground=TEXT, font=(FONT, 11, "bold"))
    st.configure("Section.TLabel", background=PANEL, foreground=ACCENT, font=(FONT, 9, "bold"))
    st.configure("TButton", padding=(6, 4), background="#e4e8ee", foreground=TEXT, bordercolor=LINE,
                 lightcolor="#eef1f5", darkcolor="#d0d6de", focuscolor=BG)
    st.map("TButton", background=[("disabled", "#eef0f3"), ("pressed", "#c9d1db"), ("active", "#d7dde5")],
           foreground=[("disabled", "#a7b0ba")])
    st.configure("Primary.TButton", background=GREEN, foreground="white", font=(FONT, 10, "bold"),
                 lightcolor=GREEN, darkcolor=GREEN)
    st.map("Primary.TButton", background=[("disabled", "#cfe3d5"), ("pressed", "#166b30"), ("active", "#23a047")],
           foreground=[("disabled", "#f4f8f5")])
    st.configure("Jog.TButton", font=(FONT, 12, "bold"), padding=(2, 1), width=3)
    st.configure("Seg.TRadiobutton", background="#e4e8ee", foreground=TEXT, padding=(8, 4), anchor="center",
                 indicatorcolor="#e4e8ee", indicatorrelief="flat", indicatormargin=0, indicatordiameter=0)
    st.map("Seg.TRadiobutton", background=[("selected", ACCENT), ("active", "#d7dde5")],
           foreground=[("selected", "white")])
    st.layout("Seg.TRadiobutton", [("Radiobutton.padding", {"sticky": "nswe", "children": [
        ("Radiobutton.label", {"sticky": "nswe"})]})])
    st.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 4, 0, 0))
    st.configure("TNotebook.Tab", padding=(11, 5), background="#e4e8ee", foreground=MUTED, font=(FONT, 10, "bold"))
    st.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", ACCENT)])
    st.configure("TCheckbutton", background=PANEL)
    st.configure("Horizontal.TScale", background=BG, troughcolor="#dfe4ea")
    st.configure("Panel.Horizontal.TScale", background=PANEL)
    st.configure("Joint.Horizontal.TProgressbar", troughcolor="#e6eaef", background=ACCENT, thickness=8,
                 bordercolor="#e6eaef", lightcolor=ACCENT, darkcolor=ACCENT)
    st.configure("TEntry", fieldbackground="white", padding=2)


class EStopButton(tk.Frame):
    """Big red E-STOP on a yellow backing plate, like the real thing.

    Built from a Label rather than a tk.Button: on macOS tk.Button ignores its background colour, which left white text
    on a pale button.  Labels keep their colours on every platform.  It acts on the PRESS (an e-stop must not wait for
    the release), responds to Return / space when focused, and looks pushed in while held."""

    def __init__(self, parent, command):
        super().__init__(parent, bg=ESTOP_YELLOW, padx=4, pady=4)
        self.command = command
        self.label = tk.Label(self, text="E-STOP", bg=ESTOP_RED, fg="white", font=(FONT, 17, "bold"),
                              relief="raised", bd=3, padx=8, pady=6, cursor="hand2", takefocus=1)
        self.label.pack(fill="both", expand=True)
        for w in (self, self.label):
            w.bind("<ButtonPress-1>", self._down)
            w.bind("<ButtonRelease-1>", self._up)
        self.label.bind("<Return>", lambda _e: self.command())
        self.label.bind("<space>", lambda _e: self.command())

    def _down(self, _event=None):
        self.label.configure(bg=ESTOP_RED_DOWN, relief="sunken")
        self.command()

    def _up(self, _event=None):
        self.label.configure(bg=ESTOP_RED, relief="raised")


class FieldList:
    """Two-column 'label: value' list on a panel; values wrap if they are long."""

    def __init__(self, parent, labels, row0=0, key_width=13):
        self.vars = {}
        for i, label in enumerate(labels):
            ttk.Label(parent, text=label, style="Key.TLabel", width=key_width, anchor="nw").grid(
                row=row0 + i, column=0, sticky="nw", pady=1)
            var = tk.StringVar(value="-")
            ttk.Label(parent, textvariable=var, style="Panel.TLabel", wraplength=WRAP - 70, justify="left").grid(
                row=row0 + i, column=1, sticky="w", pady=1)
            self.vars[label] = var

    def set(self, pairs):
        for label, value in pairs:
            if label in self.vars:
                self.vars[label].set(value)


def section(parent, text, row, pady=(8, 2)):
    ttk.Label(parent, text=text.upper(), style="Section.TLabel").grid(row=row, column=0, columnspan=8, sticky="w",
                                                                      pady=pady)


class TeachPendant:
    """The teach / jog window for the whole line.  `cell` is an integration.production_cell.ToyCarCell."""

    def __init__(self, root: tk.Tk, cell: ToyCarCell):
        self.root, self.cell = root, cell
        self._running, self._ticks, self._log_shown = True, 0, None
        root.title("Toy Car Line - teach pendant")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<FocusOut>", lambda _e: cell.stop_jog())
        root.bind("<KeyPress>", self._key)
        apply_style(root)
        self.robot_key = tk.StringVar(value=cell.stations[0].key)  # robot shown on the Jog and Move pages
        self.robot_key.trace_add("write", lambda *_: self._robot_changed())
        self.jog_buttons, self.move_buttons = [], []
        self._build()
        self._robot_changed()
        self.refresh()

    @property
    def station(self):
        return self.cell.by_key[self.robot_key.get()]

    # ================================================================== layout
    def _build(self):
        r = self.root
        r.columnconfigure(0, weight=1)
        r.rowconfigure(3, weight=1)
        self.banner = tk.Label(r, text="IDLE", font=(FONT, 18, "bold"), pady=6)
        self.banner.grid(row=0, column=0, sticky="ew")
        self.help_var = tk.StringVar(value="")
        ttk.Label(r, textvariable=self.help_var, wraplength=WRAP + 70, justify="left", style="Muted.TLabel",
                  padding=(10, 4, 10, 2)).grid(row=1, column=0, sticky="ew")
        self._build_controls(r)
        self.pages = ttk.Notebook(r)
        self.pages.grid(row=3, column=0, sticky="nsew", padx=8, pady=(4, 8))
        self.page = {}
        for name in ("Line", "Jog", "Move", "Safety", "Log"):
            frame = ttk.Frame(self.pages, style="Panel.TFrame", padding=(10, 6, 10, 10))
            frame.columnconfigure(1, weight=1)
            self.pages.add(frame, text=name)
            self.page[name] = frame
        self._build_line_page(self.page["Line"])
        self._build_jog_page(self.page["Jog"])
        self._build_move_page(self.page["Move"])
        self._build_safety_page(self.page["Safety"])
        self._build_log_page(self.page["Log"])
        self.pages.bind("<<NotebookTabChanged>>", lambda _e: (self.cell.stop_jog(), self._refresh_soon()))

    def _build_controls(self, parent):
        """Always visible: E-STOP, Start / Re-arm, the 1-2-3 recovery sequence and the simulation speed."""
        box = ttk.Frame(parent, padding=(8, 4, 8, 0))
        box.grid(row=2, column=0, sticky="ew")
        for c in range(3):
            box.columnconfigure(c, weight=1, uniform="btn")
        self.btn_start = ttk.Button(box, text="Start production", style="Primary.TButton",
                                    command=self.cell.start_production)
        self.btn_start.grid(row=0, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        self.btn_rearm = ttk.Button(box, text="Re-arm", command=self.cell.rearm)
        self.btn_rearm.grid(row=0, column=2, sticky="ew", padx=2, pady=2)
        self.btn_release = ttk.Button(box, text="1 Release", command=self.cell.safety.release_gui_estop)
        self.btn_reset = ttk.Button(box, text="2 Reset", command=self.cell.safety.request_reset)
        self.btn_resume = ttk.Button(box, text="3 Resume", command=self.cell.safety.request_start)
        for k, b in enumerate((self.btn_release, self.btn_reset, self.btn_resume)):
            b.grid(row=1, column=k, sticky="ew", padx=2, pady=2)
        self.btn_estop = EStopButton(box, self.cell.safety.press_gui_estop)
        self.btn_estop.grid(row=0, column=3, rowspan=2, sticky="nsew", padx=(8, 0), pady=2)
        speed = ttk.Frame(box)
        speed.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Label(speed, text="Sim speed", font=(FONT, 10, "bold")).pack(side="left")
        self.sim_speed = tk.DoubleVar(value=SIM_SPEEDS.index(1.0))
        ttk.Scale(speed, from_=0, to=len(SIM_SPEEDS) - 1, variable=self.sim_speed,
                  command=self._set_sim_speed).pack(side="left", padx=8, fill="x", expand=True)
        self.sim_speed_txt = tk.StringVar(value="x1")
        ttk.Label(speed, textvariable=self.sim_speed_txt, width=5, font=(FONT, 11, "bold")).pack(side="left")

    # ------------------------------------------------------------------ Line page
    def _build_line_page(self, p):
        section(p, "Production", 0, pady=(2, 3))
        labels = [label for label, _v in self.cell.line_fields() if label not in SAFETY_FIELDS]
        self.line_fields = FieldList(p, labels, row0=1, key_width=15)
        r = 1 + len(labels)
        section(p, "Robots", r)
        self.robot_rows = {}
        for k, st in enumerate(self.cell.stations):
            row = ttk.Frame(p, style="Panel.TFrame")
            row.grid(row=r + 1 + k, column=0, columnspan=2, sticky="ew", pady=2)
            dot = tk.Canvas(row, width=12, height=12, bg=PANEL, highlightthickness=0)
            dot.create_oval(1, 1, 11, 11, fill=GREY, outline="", tags="dot")
            dot.pack(side="left", padx=(0, 6))
            name = ttk.Label(row, text=f"{st.number} {st.name}", style="Value.TLabel", width=11, cursor="hand2")
            name.pack(side="left")
            name.bind("<Button-1>", lambda _e, key=st.key: (self.robot_key.set(key), self.pages.select(1)))
            var = tk.StringVar(value="-")
            ttk.Label(row, textvariable=var, style="Panel.TLabel", wraplength=WRAP - 60, justify="left").pack(
                side="left", fill="x")
            self.robot_rows[st.key] = (dot, var)
        ttk.Label(p, text="green working, grey at rest, amber held by an obstacle, red obstacle placed.  "
                          "Click a robot's name to jog it.", style="PanelMuted.TLabel", wraplength=WRAP + 40,
                  justify="left").grid(
            row=r + 6, column=0, columnspan=2, sticky="w", pady=(6, 0))

    # ------------------------------------------------------------------ robot picker (Jog + Move pages)
    def _robot_picker(self, p, row):
        bar = ttk.Frame(p, style="Panel.TFrame")
        bar.grid(row=row, column=0, columnspan=8, sticky="ew", pady=(2, 4))
        for k, st in enumerate(self.cell.stations):
            bar.columnconfigure(k, weight=1, uniform="robot")
            ttk.Radiobutton(bar, text=f"{st.number} {st.name}", value=st.key, variable=self.robot_key,
                            style="Seg.TRadiobutton").grid(row=0, column=k, sticky="ew", padx=1)
        title = tk.StringVar()
        ttk.Label(p, textvariable=title, style="PanelMuted.TLabel", wraplength=WRAP + 40, justify="left").grid(
            row=row + 1, column=0, columnspan=8, sticky="w")
        return title

    def _hold_button(self, parent, text, kind, index, sign):
        b = ttk.Button(parent, text=text, style="Jog.TButton")
        b.bind("<ButtonPress-1>", lambda _e: self._jog_press(b, kind, index, sign))
        b.bind("<ButtonRelease-1>", lambda _e: self.cell.stop_jog())
        b.bind("<Leave>", lambda _e: self.cell.stop_jog())
        self.jog_buttons.append(b)
        return b

    def _jog_press(self, button, kind, index, sign):
        if not button.instate(["disabled"]):
            self.cell.start_jog(self.robot_key.get(), kind, index, sign)

    # ------------------------------------------------------------------ Jog page
    def _build_jog_page(self, p):
        self.jog_title = self._robot_picker(p, 0)
        sp = ttk.Frame(p, style="Panel.TFrame")
        sp.grid(row=2, column=0, columnspan=8, sticky="ew", pady=(6, 0))
        ttk.Label(sp, text="Jog speed", style="Panel.TLabel").pack(side="left")
        self.jog_speed = tk.DoubleVar(value=self.cell.jog_speed * 100)
        ttk.Scale(sp, from_=10, to=100, variable=self.jog_speed, style="Panel.Horizontal.TScale",
                  command=lambda _v: setattr(self.cell, "jog_speed", float(self.jog_speed.get()) / 100)).pack(
            side="left", padx=6, fill="x", expand=True)
        self.jog_speed_txt = tk.StringVar()
        ttk.Label(sp, textvariable=self.jog_speed_txt, style="PanelMuted.TLabel", width=17).pack(side="left")

        section(p, "Joint jog  (hold to move)", 3)
        jj = ttk.Frame(p, style="Panel.TFrame")
        jj.grid(row=4, column=0, columnspan=8, sticky="ew")
        jj.columnconfigure(3, weight=1)
        self.joint_vars, self.joint_bars, self.joint_lims = [], [], []
        for j in range(6):
            ttk.Label(jj, text=f"J{j + 1}", style="Value.TLabel", width=3).grid(row=j, column=0, sticky="w")
            self._hold_button(jj, "−", "joint", j, -1).grid(row=j, column=1, padx=1, pady=1)
            self._hold_button(jj, "+", "joint", j, +1).grid(row=j, column=2, padx=1, pady=1)
            bar = ttk.Progressbar(jj, maximum=100, style="Joint.Horizontal.TProgressbar")
            bar.grid(row=j, column=3, sticky="ew", padx=(8, 6))
            var = tk.StringVar(value="0.0")
            ttk.Label(jj, textvariable=var, style="Value.TLabel", width=7, anchor="e").grid(row=j, column=4)
            lim = tk.StringVar(value="")
            ttk.Label(jj, textvariable=lim, style="PanelMuted.TLabel", width=9, anchor="e").grid(row=j, column=5)
            self.joint_vars.append(var)
            self.joint_bars.append(bar)
            self.joint_lims.append(lim)

        section(p, "Cartesian jog  (world frame, hold to move)", 5)
        cj = ttk.Frame(p, style="Panel.TFrame")
        cj.grid(row=6, column=0, columnspan=8, sticky="w")
        for a, name in enumerate(AXES):
            ttk.Label(cj, text=name, style="Value.TLabel").grid(row=0, column=3 * a, padx=(0 if a == 0 else 14, 4))
            self._hold_button(cj, "−", "cart", a, -1).grid(row=0, column=3 * a + 1, padx=1)
            self._hold_button(cj, "+", "cart", a, +1).grid(row=0, column=3 * a + 2, padx=1)

        self.jog_readout = FieldList(p, ("Tool point", "Doing"), row0=7)

    # ------------------------------------------------------------------ Move page
    def _build_move_page(self, p):
        self.move_title = self._robot_picker(p, 0)
        section(p, "Go to joint state (degrees)", 2)
        g = ttk.Frame(p, style="Panel.TFrame")
        g.grid(row=3, column=0, columnspan=8, sticky="ew")
        self.goto_entries = []
        for j in range(6):
            g.columnconfigure(j, weight=1, uniform="q")
            ttk.Label(g, text=f"J{j + 1}", style="PanelMuted.TLabel").grid(row=0, column=j)
            e = ttk.Entry(g, width=6, justify="right")
            e.grid(row=1, column=j, padx=1, sticky="ew")
            self.goto_entries.append(e)
        b = ttk.Frame(p, style="Panel.TFrame")
        b.grid(row=4, column=0, columnspan=8, sticky="ew", pady=(6, 0))
        buttons = (("Go", self._goto, True), ("Home", lambda: self.cell.go_home(self.robot_key.get()), True),
                   ("Copy current", self._copy_current, False),
                   ("Store as final", lambda: self.cell.store_final(self.robot_key.get()), False),
                   ("Go to final", self._go_final, True), ("Show final", self._show_final, False))
        for k, (text, cmd, gated) in enumerate(buttons):
            b.columnconfigure(k % 3, weight=1, uniform="mv")
            w = ttk.Button(b, text=text, command=cmd, style="Primary.TButton" if text == "Go" else "TButton")
            w.grid(row=k // 3, column=k % 3, sticky="ew", padx=1, pady=1)
            if gated:
                self.move_buttons.append(w)

        ob = ttk.Frame(p, style="Panel.TFrame")
        ob.grid(row=5, column=0, columnspan=8, sticky="ew", pady=(4, 0))
        ob.columnconfigure((0, 1), weight=1)
        ttk.Button(ob, text="Drop test obstacle in path",
                   command=lambda: self.cell.set_obstacle(self.robot_key.get(), True)).grid(
            row=0, column=0, sticky="ew", padx=1)
        ttk.Button(ob, text="Remove obstacle", command=lambda: self.cell.set_obstacle(self.robot_key.get(), False)).grid(
            row=0, column=1, sticky="ew", padx=1)

        section(p, "Robot details", 7)
        self.detail_box = ttk.Frame(p, style="Panel.TFrame")
        self.detail_box.grid(row=8, column=0, columnspan=8, sticky="ew")
        self.detail_vars, self._detail_key = [], None

    def _rebuild_details(self):
        """The detail rows differ per robot (each station adds its own), so they are rebuilt when the robot changes."""
        for w in self.detail_box.winfo_children():
            w.destroy()
        fields = self.cell.robot_fields(self.station)
        self.details = FieldList(self.detail_box, [label for label, _v in fields if label not in JOG_PAGE_FIELDS],
                                 key_width=18)
        self._detail_key = self.station.key

    # ------------------------------------------------------------------ Safety page
    def _build_safety_page(self, p):
        c = self.cell
        section(p, "Person", 0, pady=(2, 3))
        pb = ttk.Frame(p, style="Panel.TFrame")
        pb.grid(row=1, column=0, columnspan=2, sticky="ew")
        pb.columnconfigure((0, 1), weight=1)
        ttk.Button(pb, text="Enter: loading gate", command=lambda: self._person_in("load")).grid(
            row=0, column=0, sticky="ew", padx=1, pady=1)
        ttk.Button(pb, text="Enter: sorting door", command=lambda: self._person_in("sort")).grid(
            row=0, column=1, sticky="ew", padx=1, pady=1)
        ttk.Button(pb, text="Person leaves", command=c.person.go_outside).grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=1, pady=1)
        section(p, "Light curtain", 2)
        self.curtain_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(p, text="Force loading-gate light curtain broken", variable=self.curtain_var,
                        command=lambda: setattr(c, "force_load_curtain", self.curtain_var.get())).grid(
            row=3, column=0, columnspan=2, sticky="w")
        section(p, "Safety inputs", 4)
        self.safety_fields = FieldList(p, SAFETY_FIELDS, row0=5)
        section(p, "Keyboard (pendant focused)", 10)
        keys = ttk.Frame(p, style="Panel.TFrame")
        keys.grid(row=11, column=0, columnspan=2, sticky="w")
        for k, (key, what) in enumerate((("x", "e-stop"), ("z", "person in / out"), ("c", "change door"),
                                         ("1-4", "pick a robot"), ("u / i", "MH5 obstacle on / off"),
                                         ("j / h", "MPX3500 obstacle"), ("o / p", "FANUC obstacle"),
                                         ("k / l", "UR3e obstacle"))):
            ttk.Label(keys, text=key, style="Value.TLabel", width=5).grid(row=k // 2, column=2 * (k % 2), sticky="w")
            ttk.Label(keys, text=what, style="PanelMuted.TLabel", width=20).grid(row=k // 2, column=2 * (k % 2) + 1,
                                                                                  sticky="w")

    # ------------------------------------------------------------------ Log page
    def _build_log_page(self, p):
        p.rowconfigure(0, weight=1)
        p.columnconfigure(0, weight=1)
        self.log = tk.Text(p, height=12, width=46, state="disabled", font=("Courier", 9), wrap="word", bg=PANEL,
                           fg=TEXT, relief="flat", highlightthickness=1, highlightbackground=LINE)
        sb = ttk.Scrollbar(p, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

    # ================================================================ commands
    def _robot_changed(self):
        self.cell.stop_jog()
        st = self.station
        lo, hi = st.arm.robot.qlim
        for lim, l, h in zip(self.joint_lims, lo, hi):
            lim.set(f"{np.degrees(l):.0f}..{np.degrees(h):.0f}")
        for e, qd in zip(self.goto_entries, np.degrees(st.home_q)):
            e.delete(0, "end")
            e.insert(0, f"{qd:.1f}")
        title = f"Robot {st.number}: {st.model}  -  {st.title}  ({st.owner})"
        self.jog_title.set(title)
        self.move_title.set(title)
        self._rebuild_details()
        self._refresh_soon()

    def _refresh_soon(self):
        if hasattr(self, "log"):
            self.root.after_idle(self.refresh)

    def _goto(self):
        try:
            q = np.radians([float(e.get()) for e in self.goto_entries])
        except ValueError:
            self.cell.say(f"{self.station.name}: go to - enter six numbers (degrees)")
            return
        self.cell.goto(self.robot_key.get(), q)

    def _go_final(self):
        st = self.station
        self.cell.goto(st.key, st.final_q, "moving to the FINAL joint state")

    def _show_final(self):
        """Put the stored final pose in the boxes (to check or edit it before Go)."""
        for e, qd in zip(self.goto_entries, np.degrees(self.station.final_q)):
            e.delete(0, "end")
            e.insert(0, f"{qd:.1f}")

    def _copy_current(self):
        for e, qd in zip(self.goto_entries, np.degrees(self.station.arm.q)):
            e.delete(0, "end")
            e.insert(0, f"{qd:.1f}")

    def _person_in(self, door):
        if not self.cell.person.go_inside(door):
            self.cell.say("The person can only change door while outside the cell")

    def _set_sim_speed(self, _value=None):
        """The slider snaps to the steps in SIM_SPEEDS (slow motion below x1, fast-forward above)."""
        k = int(round(float(self.sim_speed.get())))
        self.sim_speed.set(k)
        self.cell.sim_speed = SIM_SPEEDS[k]
        self.sim_speed_txt.set(f"x{SIM_SPEEDS[k]:g}")

    def _key(self, event):
        if isinstance(event.widget, (tk.Entry, ttk.Entry)):
            return  # typing a joint angle
        k, c = event.char.lower(), self.cell
        if k == "x":
            c.safety.press_gui_estop()
        elif k == "z":
            c.person.toggle()
        elif k == "c":
            if c.person.cycle_door():
                c.say(f"Person is now at the {c.person.status}")
        elif k in KEY_OBSTACLES:
            c.set_obstacle(*KEY_OBSTACLES[k])
        elif k in "1234" and k:
            self.robot_key.set(c.stations[int(k) - 1].key)

    # ================================================================= display
    def refresh(self):
        c, s = self.cell, self.cell.safety
        state, extra = c.banner()
        bg, fg = STATE_COLOURS.get(state, ("#5b6068", "white"))
        self.banner.configure(text=state + (f"  –  {extra}" if extra else ""), bg=bg, fg=fg,
                              wraplength=WRAP + 80)
        self.help_var.set(STATE_HELP.get(state, ""))
        clear = not s.active and not c.zone_causes() and not s.gui_estop
        self._enable(self.btn_start, c.production_state == "not started" and s.state != "STOPPED" and clear)
        self._enable(self.btn_rearm, c.production_state in ("complete", "FAULT") and s.state != "STOPPED" and clear)
        self._enable(self.btn_release, s.gui_estop)
        self._enable(self.btn_reset, s.state == "STOPPED" and clear)
        self._enable(self.btn_resume, s.state == "READY" and c.production_running and clear)
        page = self.pages.tab(self.pages.select(), "text")
        if page == "Line":
            self.line_fields.set(c.line_fields())
            for st in c.stations:
                dot, var = self.robot_rows[st.key]
                text = st.activity
                at_rest = text.startswith(("idle", "at FINAL"))
                colour = AMBER if st.blocked else RED if st.obstacle_active else GREY if at_rest else GREEN
                dot.itemconfigure("dot", fill=colour)
                var.set(text + ("   [obstacle]" if st.obstacle_active else ""))
        elif page == "Jog":
            self._refresh_jog(c.manual_allowed)
        elif page == "Move":
            manual = c.manual_allowed
            for b in self.move_buttons:
                self._enable(b, manual)
            if self._detail_key != self.station.key:
                self._rebuild_details()
            self.details.set(c.robot_fields(self.station))
        elif page == "Safety":
            self.safety_fields.set(c.line_fields())
        elif page == "Log":
            self._refresh_log()

    def _refresh_jog(self, manual):
        c, st = self.cell, self.station
        lo, hi = st.arm.robot.qlim
        for var, bar, qj, l, h in zip(self.joint_vars, self.joint_bars, st.arm.q, lo, hi):
            var.set(f"{np.degrees(qj):.1f}°")
            bar["value"] = 100 * (qj - l) / (h - l)
        self.jog_speed_txt.set(f"{np.degrees(JOG_JOINT_SPEED * c.jog_speed):.0f}°/s  "
                               f"{1000 * JOG_CART_SPEED * c.jog_speed:.0f} mm/s")
        self.jog_readout.set([("Tool point", "{:.3f}, {:.3f}, {:.3f} m".format(*st.arm.tcp())),
                              ("Doing", st.activity if manual else f"{st.activity}  (jog locked)")])
        for b in self.jog_buttons:
            self._enable(b, manual)

    def _refresh_log(self):
        events = self.cell.events
        last = events[-1] if events else None
        if last != self._log_shown:
            self.log.configure(state="normal")
            self.log.delete("1.0", "end")
            self.log.insert("end", "\n".join(list(events)[-120:]))
            self.log.see("end")
            self.log.configure(state="disabled")
            self._log_shown = last

    @staticmethod
    def _enable(widget, on):
        widget.state(["!disabled"] if on else ["disabled"])

    # ==================================================================== loop
    def start_loop(self):
        self._schedule(1)

    def _schedule(self, delay_ms):
        if self._running:
            self._after_id = self.root.after(delay_ms, self._step)

    def _step(self):
        if not self._running:
            return
        t0 = time.perf_counter()
        self.cell.tick()
        self._ticks += 1
        if self._ticks % REFRESH_EVERY == 0:
            self.refresh()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._schedule(max(1, int(DT * 1000 - elapsed_ms)))

    def close(self):
        self._running = False
        self.cell.stop_jog()
        if getattr(self, "_after_id", None) is not None:
            self.root.after_cancel(self._after_id)
        self.root.destroy()


def run_gui(cell: ToyCarCell, root: tk.Tk | None = None) -> None:
    """Open the teach pendant for `cell` and run until the window is closed.  It opens at the right-hand edge of the
    screen so the Swift view can sit beside it."""
    root = root or tk.Tk()
    app = TeachPendant(root, cell)
    root.deiconify()  # ProductionLine.py creates the window hidden, before the scene is built
    root.update_idletasks()
    w, h = root.winfo_reqwidth(), root.winfo_reqheight()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.minsize(w, min(h, sh - 80))
    root.geometry(f"+{max(0, sw - w - 12)}+{max(0, min(40, sh - h - 60))}")
    root.lift()
    root.attributes("-topmost", True)
    root.after(800, lambda: root.attributes("-topmost", False))
    app.start_loop()
    root.mainloop()
