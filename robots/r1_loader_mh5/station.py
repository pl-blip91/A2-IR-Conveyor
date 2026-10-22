import numpy as np

from shared import line_config as cfg
from shared.motion import Arm
from shared.station_base import RobotStation

from . import loading_config as lc
from .loading_station import LoadingStation, RunState

MH5_EPS = 0.02  # manipulability below which RMRC damping switches on (same threshold, damps below)
MH5_STEP = np.deg2rad(2.0)  # largest joint step per frame in planned joint moves


class LoaderArm(Arm):
    """The MH5 on the TurtleBot as an `Arm`, checked with capsule envelope and spherical obstacle."""

    def __init__(self, station):
        super().__init__("MH5", station.arm, station.obstacle, 0.0, MH5_EPS, MH5_STEP)
        self.station = station
        self.safe_q = lc.Q_TRAVEL.copy()

    def hits(self, q_list):
        st = self.station
        return bool(st.obstacle.active and len(q_list) > 0
                    and st.checker.path_hit(np.asarray(q_list, float), stride=1, with_statics=False) is not None)

    def jog_blocked(self, q_new):
        """Jogging also keeps the arm out of the belt and the supply table (Lachlan's static keep-outs)."""
        what = self.station.checker.config_hit(np.asarray(q_new, float), with_statics=True)
        if what == "obstacle":
            return "obstacle in the way"
        if what is not None:
            return "the arm would hit the belt / supply table"
        return super().jog_blocked(q_new)


class LoaderStation(RobotStation):
    number, key, name = 1, "mh5", "MH5"
    title = "Loading"
    model = "Yaskawa Motoman MH5 on a TurtleBot3 Waffle"
    owner = "Lachlan"

    def __init__(self, env, cars):
        self.loading = LoadingStation(env, cars, dt=cfg.DT)
        self.arm = LoaderArm(self.loading)
        self.home_q = lc.Q_TRAVEL.copy()
        self.final_q = lc.Q_TRAVEL.copy()

    # ------------------------------------------------------------------ production
    def run(self, line):
        """Load every car onto the belt.  The conveyor's own slot photo-eyes report each car as it is put down."""
        line.set_stage("loading", key="load")
        yield from self.loading.job()

    # ------------------------------------------------------------------ line services
    def draw(self):
        self.loading.sync()

    @property
    def obstacle_active(self):
        return bool(self.loading.obstacle.active)

    def set_obstacle(self, on):
        st = self.loading
        if not on:
            if st.obstacle.active:
                st.remove_obstacle()
        elif not st.obstacle.active:
            if st.drop_obstacle_in_path() is None:  # nothing is moving: park it just above the belt at the dock
                st.place_obstacle([cfg.LOAD_X, 0.0, cfg.BELT_HEIGHT + 0.25])

    @property
    def blocked(self):
        return bool(self.arm.blocked or self.loading.blocked)

    @property
    def replans(self):
        return self.arm.replans + self.loading.replans

    @property
    def phase(self):
        return self.loading.phase

    @property
    def activity(self):
        """While the loading job runs, its phase says what the robot is doing (the arm note only covers manual moves)."""
        if self.loading.run_state in (RunState.RUNNING, RunState.BLOCKED):
            return self.loading.phase
        return self.arm.note

    def info(self):
        s = self.loading.status()
        bx, by, byaw = s["base_xy_yaw"]
        m, e = s["manipulability"], s["vision_error_px"]
        return [("Phase", s["phase"]),
                ("Cars placed", f"{s['cars_placed']} / {self.loading.n_cars}" + ("  (holding a car)" if s["holding_car"] else "")),
                ("Base x, y, yaw", f"{bx:.2f}, {by:.2f} m, {np.degrees(byaw):.0f} deg"),
                ("Vision error", "-" if e != e else f"{e:.2f} px"),
                ("RMRC manipulability", "-" if m != m else f"{m:.4f}"),
                ("Blocked because", s["block_reason"] or "-")]

    def reset(self):
        self.loading.reset()
        self.arm.note = "idle"
