import time
from collections import deque

import numpy as np

from shared import line_config as cfg
from shared.car import Car, CarProfile
from shared.conveyor import Conveyor
from shared.factory import build_factory
from shared.motion import jog_cart_step, jog_joint_step, plan_move
from shared.safety import (READY, RUNNING, STOPPED, SLOW_SCALE, CURTAIN_ABS_Y, HardwareEstop, LightCurtain, Person,
                           SafetySystem, StackLight, build_safety_cell)
from shared.sensors import BeltSlotSensors
from shared.static_mesh import StaticScene

from robots.r1_loader_mh5.station import LoaderStation
from robots.r2_painter_mpx3500.station import PaintingStation
from robots.r3_stamper_fanuc.station import StampingStation
from robots.r4_sorter_ur3e.station import SortingStation

DT = cfg.DT
SLOT_WAIT_S = 3.0  # the belt waits this long for the slot photo-eyes to report every car before it faults
JOG_JOINT_SPEED = np.deg2rad(20)  # rad/s at 100 % jog speed (hold-to-move)
JOG_CART_SPEED = 0.10  # m/s at 100 % jog speed


class ToyCarCell:
    def __init__(self, env, factory=True, gamepad=True):
        self.env = env
        env.set_camera_pose(cfg.CAMERA_POSITION, cfg.CAMERA_LOOK_AT)
        self.events = deque(maxlen=200)
        self.time = 0.0
        self.message = ""
        static = StaticScene("cell")  # everything that never moves is merged into a few meshes (see static_mesh.py)
        if factory:
            build_factory(static)
        # --- conveyor (Aarav's belt for the whole line) and its slot photo-eyes ---
        self.conveyor = Conveyor(env)
        self.slot_sensors = BeltSlotSensors(env)
        # --- the three toy cars (one STL, grey until painted); blue goes first, then red, then green ---
        self.profile = CarProfile()
        self.cars = [Car(env, name) for name in cfg.CAR_ORDER]
        self.order = list(self.cars)
        # --- the four robots, in line order ---
        self.loader = LoaderStation(env, self.cars)
        self.painter = PaintingStation(env, self.profile)
        self.stamper = StampingStation(env, self.profile)
        self.sorter = SortingStation(env)
        self.stations = [self.loader, self.painter, self.stamper, self.sorter]
        self.by_key = {s.key: s for s in self.stations}
        self.by_name = {s.name: s for s in self.stations}
        self.loader.loading.on_log = lambda text: self.say(f"MH5: {text}")
        # --- safety equipment + state ---
        self.estops, (door_l, door_r), (g0, g1) = build_safety_cell(env, static, cfg.STAMP_X, cfg.SORT_X)
        self.curtains = {"sort": LightCurtain(env, door_l + 0.10, door_r - 0.10, cfg.SORT_SIDE * CURTAIN_ABS_Y),
                         "load": LightCurtain(env, g0 + 0.10, g1 - 0.10, +CURTAIN_ABS_Y)}
        self.towers = [StackLight(env, door_r + 0.35, cfg.SORT_SIDE * (cfg.FENCE_Y + 0.35)),
                       StackLight(env, g0 - 0.30, cfg.FENCE_Y + 0.35)]
        self.person = Person(env)
        self.force_load_curtain = False  # simulated stuck-on light-curtain signal (Lachlan's 'force curtain broken')
        self.safety = SafetySystem()
        self.hw = HardwareEstop(enabled=gamepad)
        static.add_to(env)
        # --- production / manual control ---
        self.job = None  # the production generator (advanced only while RUNNING)
        self.production_state = "not started"
        self.stage, self.stage_key = "idle", "idle"
        self.manual = {}  # station key -> generator of a manual move (go to / home / final pose)
        self.jog = None  # (station key, "joint" | "cart", index, sign) while a jog button is held
        self.jog_speed = 0.5  # fraction of JOG_*_SPEED
        self._jog_msg = None
        self.cmds = []  # requests from the terminal thread, drained in the main loop
        self.sim_speed, self.speed_acc, self.frame = 1.0, 0.0, 0  # sim_speed: set by the pendant's slider
        self._belt_moved = False
        self._printed = None
        self._last_held = {}
        self.follow()
        env.step()
        # The MPX3500 spray paths are planned ONCE here (a few seconds of IK) so nothing slow ever runs while the line is
        # live and the e-stop has to be answered every frame.
        self.painter.prepare()
        self.follow()
        env.step()
        self.say("Scene ready. Press 'Start production', or jog any robot from the Jog page.")

    # =========================================================================
    # messages
    # =========================================================================
    def say(self, text):
        self.message = text
        self.events.append(f"[{self.time:7.1f}s] {text}")
        print(text)

    def set_stage(self, text, key=None):
        """Called by the stations' run() to say what the line is doing (shown on the pendant, not logged)."""
        self.stage = text
        self.stage_key = key or text

    def log_robot_changes(self):
        """Once per frame: log a robot being held by an obstacle, and let go again (changes only).  The rest of the
        event log is kept short on purpose: sensor detections, the loader's alignment, one line per robot per car
        (the stations log those themselves), safety and production events."""
        for st in self.stations[1:]:  # (the MH5's loading job logs its own holds)
            held = st.blocked
            if held != self._last_held.get(st.key, False):
                self.say(f"{st.name}: " + ("HELD - obstacle in its path" if held else "path clear, carrying on"))
            self._last_held[st.key] = held

    # =========================================================================
    # safety inputs and scenery
    # =========================================================================
    def curtain_tripped(self, key):
        if key == "load" and self.force_load_curtain:
            return True
        return self.person.door == key and self.person.in_curtain()

    def zone_causes(self):
        """Every zone input that is active this frame (each one is a latched stop cause)."""
        causes = []
        if self.force_load_curtain:
            causes.append(f"LIGHT CURTAIN ({cfg.DOORS['load']['name']}, forced)")
        if self.person.in_curtain():
            causes.append(f"LIGHT CURTAIN ({cfg.DOORS[self.person.door]['name']})")
        if self.person.in_load_zone():
            causes.append("AREA SCANNER (loading zone)")
        return causes

    def speed_scale(self):
        return SLOW_SCALE if self.person.in_warning_zone() else 1.0

    def lamp_state(self):
        """Stack-light colour: red = e-stop or fault, amber = ready / protective stop / holding, green = running."""
        s = self.safety
        if self.production_state == "FAULT":
            return STOPPED
        if s.state == STOPPED:
            return STOPPED if s.estopped else READY
        if s.state == RUNNING:
            return READY if any(st.blocked for st in self.stations) else RUNNING
        return READY

    def update_scenery(self):
        for key, curtain in self.curtains.items():
            curtain.set(self.curtain_tripped(key))
        lamp = self.lamp_state()
        for tower in self.towers:
            tower.set(lamp)
        pressed = self.safety.gui_estop or self.safety.hw_pressed
        for post in self.estops:
            post.set(pressed)

    # =========================================================================
    # one frame
    # =========================================================================
    @property
    def production_running(self):
        return self.production_state == "running"

    @property
    def manual_allowed(self):
        """Manual control (jog, go to, home, final pose) is allowed when no stop is latched or active and production is
        not in progress (not started yet, complete, or aborted by a fault).  Hold-to-move jogging works in READY too:
        the jog button is a hold-to-run control, like the enabling switch on a real pendant."""
        return (self.safety.state != STOPPED and not self.safety.active and not self.zone_causes()
                and not self.safety.gui_estop and self.production_state != "running")

    @property
    def conveyor_moving(self):
        return self._belt_moved

    def tick(self):
        self.frame += 1
        self.time += DT
        self.person.update(DT)  # the person moves whatever the cell is doing
        if self.safety.update(self.hw.pressed(), self.zone_causes()):
            self.on_stop()
        if self.safety.message != self._printed:
            self._printed = self.safety.message
            self.say(self.safety.message)
        self.process_commands()
        self.scan_sensors()
        s0 = self.conveyor.s
        self.speed_acc += self.speed_scale() * self.sim_speed  # 50 % speed = advance one step every second frame
        steps = int(self.speed_acc)
        self.speed_acc -= steps
        running = self.safety.state == RUNNING
        for _ in range(steps):
            if running and self.job is not None:
                self.advance_production()
            if self.manual and self.manual_allowed:
                self.advance_manual()
        if self.jog is not None:
            self.jog_step()
        self._belt_moved = abs(self.conveyor.s - s0) > 1e-9
        self.log_robot_changes()
        self.painter.spray_live = running
        self.follow()
        self.update_scenery()
        self.env.step(DT)

    def scan_sensors(self):
        """Input scan: every sensor is read every frame, whatever the cell is doing."""
        for st in self.stations:
            if st.sensor is not None:
                st.sensor.update(self.cars)
        self.slot_sensors.update(self.cars)
        for k, car in enumerate(self.slot_sensors.car_at):  # the belt takes on a car only once its photo-eye sees it
            if car is not None and not self.conveyor.rides(car):
                self.conveyor.attach(car)
                self.say(f"Conveyor: slot {k + 1} sensor detected the {car.colour} car")

    def follow(self):
        """Everything attached to an arm follows it (shape.T = fkine(q) @ ...)."""
        for st in self.stations:
            st.draw()

    def on_stop(self):
        """Known safe state: jogging and manual moves are cancelled; production stays paused exactly where it is."""
        self.stop_jog()
        for key, gen in list(self.manual.items()):
            gen.close()
            self.by_key[key].arm.note = "idle (manual move cancelled by the stop)"
        self.manual.clear()

    def advance_production(self):
        try:
            next(self.job)
        except StopIteration:
            self.job = None
        except RuntimeError as e:  # e.g. no IK solution / sensor timeout: report once, stop commanding, keep running
            self.say(f"SOFTWARE FAULT: {e}")
            self.production_state = "FAULT"
            self.job = None

    def advance_manual(self):
        for key, gen in list(self.manual.items()):
            try:
                next(gen)
            except StopIteration:
                del self.manual[key]
            except RuntimeError as e:
                self.say(f"{self.by_key[key].name}: {e}")
                del self.manual[key]

    def run(self):
        """Main loop without the teach pendant (paced to DT per frame)."""
        while True:
            t0 = time.perf_counter()
            self.tick()
            spare = DT - (time.perf_counter() - t0)
            if spare > 0:
                time.sleep(spare)

    # =========================================================================
    # production
    # =========================================================================
    def move_all(self, targets, note):
        """Generator: move several arms to their target poses at the same time (each with the planner's look-ahead)."""
        gens = {}
        for st, q in targets:
            if not np.allclose(st.arm.q, q, atol=1e-6):
                gens[st.key] = plan_move(st.arm, q, note)
        while gens:
            for key in list(gens):
                try:
                    next(gens[key])
                except StopIteration:
                    del gens[key]
            yield

    def wait_for_slot_sensors(self):
        """The belt does not start until the photo-eye of every loading slot reports a car."""
        self.set_stage("belt waiting for the slot photo-eyes to see every car")
        for _ in range(int(SLOT_WAIT_S / DT)):
            if self.slot_sensors.all_present:
                self.say("Conveyor: all 3 cars detected - belt starting")
                return
            yield
        raise RuntimeError(f"belt slot photo-eyes see only {sum(self.slot_sensors.present)} of {len(self.cars)} cars")

    def production(self):
        """HOME -> LOAD -> (slot sensors) -> PAINT -> STAMP -> SORT -> every arm to its FINAL joint state."""
        self.set_stage("homing every robot before production")
        yield from self.move_all([(st, st.home_q) for st in self.stations], "homing before production")
        for st in self.stations:
            st.arm.note = "idle (home)"
        yield from self.loader.run(self)
        yield from self.wait_for_slot_sensors()
        yield from self.painter.run(self)
        yield from self.stamper.run(self)
        yield from self.sorter.run(self)
        self.set_stage("all cars sorted: robots going to their final poses")
        yield from self.move_all([(st, st.final_q) for st in self.stations], "moving to the FINAL joint state")
        for st in self.stations:
            st.arm.note = "at FINAL pose"
        self.set_stage("all cars loaded, painted, stamped and sorted, robots at their final poses")
        self.production_state = "complete"
        self.say("Production complete - manual control is available again; Re-arm to run the line again")

    def start_production(self):
        if self.production_state != "not started":
            self.say(f"Start refused: production is '{self.production_state}' - press Re-arm first")
            return False
        if self.safety.state == STOPPED or self.safety.active:
            self.say("Start refused: clear the stop, Reset, then Start")
            return False
        self.stop_jog()
        for gen in self.manual.values():
            gen.close()
        self.manual.clear()
        self.production_state = "running"
        self.job = self.production()
        self.safety.request_start()
        self.say("Production started")
        return True

    def rearm(self):
        """Put the line back to its start state for another run (after it completed, or after a fault)."""
        if self.production_state == "running":
            self.say("Re-arm refused: production is running")
            return False
        if self.safety.state == STOPPED or self.safety.active:
            self.say("Re-arm refused: clear the stop and Reset first")
            return False
        self.stop_jog()
        self.manual.clear()
        self.job = None
        self.conveyor.clear()
        for car in self.cars:
            car.reset()
        for st in self.stations:
            st.reset()  # the loader puts the cars back on its supply table
            st.arm.blocked = False
        self.production_state = "not started"
        self.set_stage("idle")
        self.say("Line re-armed: cars back on the supply table. Press 'Start production'.")
        return True

    # =========================================================================
    # manual control (teach pendant / terminal)
    # =========================================================================
    def can_move(self, st, what):
        if not self.manual_allowed:
            why = ("production is in progress" if self.production_running
                   else "a stop is latched or active (clear it and Reset)")
            self.say(f"{st.name}: {what} refused - {why}")
            return False
        return True

    def start_jog(self, key, kind, index, sign):
        st = self.by_key[key]
        if not self.can_move(st, "jog"):
            return
        if key in self.manual:  # a jog takes over from a go-to on the same robot
            self.manual.pop(key).close()
        st.arm.jog_R = None
        self.jog = (key, kind, index, sign)
        self._jog_msg = None

    def stop_jog(self):
        if self.jog is not None:
            st = self.by_key[self.jog[0]]
            st.arm.jog_R = None
            if st.arm.note.startswith("jog"):
                st.arm.note = "idle (manual)"
        self.jog = None

    def jog_step(self):
        key, kind, index, sign = self.jog
        st = self.by_key[key]
        if not self.manual_allowed:
            self.stop_jog()
            return
        scale = self.jog_speed * self.speed_scale()
        if kind == "joint":
            why = jog_joint_step(st.arm, index, sign, JOG_JOINT_SPEED * scale)
            st.arm.note = f"jog J{index + 1} {'+' if sign > 0 else '-'}"
        else:
            why = jog_cart_step(st.arm, index, sign, JOG_CART_SPEED * scale)
            st.arm.note = f"jog {'XYZ'[index]} {'+' if sign > 0 else '-'} (RMRC)"
        if why:
            st.arm.note = f"jog held: {why}"
            if why != self._jog_msg:
                self.say(f"{st.name}: jog held - {why}")
        self._jog_msg = why

    def goto(self, key, q, note="moving to the commanded joint state"):
        """Planned joint move of one robot (look-ahead + re-planning round its obstacle)."""
        st = self.by_key[key]
        q = np.asarray(q, float)
        lo, hi = st.arm.robot.qlim
        bad = [j + 1 for j in range(len(q)) if q[j] < lo[j] - 1e-9 or q[j] > hi[j] + 1e-9]
        if bad:
            self.say(f"{st.name}: joint state REJECTED - outside the joint limits of J{bad}")
            return False
        if not self.can_move(st, "move"):
            return False
        if self.jog is not None and self.jog[0] == key:
            self.stop_jog()
        if key in self.manual:
            self.manual.pop(key).close()
        self.manual[key] = self._manual_move(st, q, note)
        self.say(f"{st.name}: {note} {np.round(np.degrees(q), 1).tolist()} deg")
        return True

    def _manual_move(self, st, q, note):
        yield from plan_move(st.arm, q, note)
        st.arm.note = "idle (manual)"

    def go_home(self, key):
        return self.goto(key, self.by_key[key].home_q, "going home")

    def store_final(self, key):
        st = self.by_key[key]
        st.final_q = st.arm.q.copy()
        self.say(f"{st.name}: FINAL pose stored = {np.round(np.degrees(st.final_q), 1).tolist()} deg")

    def set_final(self, key, q, go=True):
        """Validate a final joint state against the joint limits, store it, optionally go there."""
        st = self.by_key[key]
        lo, hi = st.arm.robot.qlim
        bad = [j + 1 for j in range(len(q)) if q[j] < lo[j] or q[j] > hi[j]]
        if bad:
            self.say(f"{st.name}: final state REJECTED - outside the joint limits of J{bad}")
            return
        st.final_q = np.asarray(q, float)
        self.say(f"{st.name}: FINAL pose = {np.round(np.degrees(q), 1).tolist()} deg")
        if go:
            if self.production_running:
                self.say("Final pose stored - production will finish by moving each robot to it")
            else:
                self.goto(key, st.final_q, "moving to the FINAL joint state")

    def set_obstacle(self, key, on):
        st = self.by_key[key]
        st.set_obstacle(on)
        self.say(f"{st.name}: test obstacle {'placed in its path' if on else 'removed'}")

    def process_commands(self):
        while self.cmds:  # finite: only what has arrived since the last frame
            cmd = self.cmds.pop(0)
            if cmd[0] == "console":
                self.handle_console(cmd[1])

    def handle_console(self, line):
        """Typed command from the terminal:   final mh5|mpx|fanuc|ur d1 d2 d3 d4 d5 d6   (degrees)."""
        p = line.split()
        if len(p) == 8 and p[0] == "final" and p[1] in self.by_key:
            try:
                q = np.deg2rad([float(v) for v in p[2:]])
            except ValueError:
                self.say("final: the six joint angles must be numbers (degrees)")
                return
            self.set_final(p[1], q)
        elif p:
            print("Terminal command:  final mh5|mpx|fanuc|ur d1 d2 d3 d4 d5 d6   (degrees)")

    # =========================================================================
    # what the teach pendant shows
    # =========================================================================
    def banner(self):
        """(state word, extra text) for the pendant's state banner."""
        s = self.safety
        if self.production_state == "FAULT":
            return "FAULT", ""
        if s.state == STOPPED:
            return ("ESTOPPED" if s.estopped else "PROTECTIVE"), ", ".join(s.latched)
        if s.state == READY:
            return ("STOPPED" if self.production_running else "IDLE"), ""
        if self.production_running:
            return ("BLOCKED" if any(st.blocked for st in self.stations) else "RUNNING"), ""
        if self.production_state == "complete":
            return "DONE", ""
        return "MANUAL", ""

    def line_fields(self):
        s, p = self.safety, self.person
        stage = f"loading: {self.loader.phase}" if self.stage_key == "load" else self.stage
        curt = ", ".join(f"{cfg.DOORS[k]['name']}: " + ("BROKEN" if self.curtain_tripped(k) else "clear")
                         for k in self.curtains)
        c = self.sorter.counts
        return [
            ("Production", self.production_state),
            ("Stage", stage),
            ("Cars", f"loaded {self.loader.loading.placed}  painted {self.painter.painted}  "
                     f"stamped {self.stamper.stamped}  sorted {self.sorter.sorted} / {len(self.cars)}"),
            ("Boxes", f"blue {c['blue']}   red {c['red']}   green {c['green']}"),
            ("Belt", f"{'MOVING' if self.conveyor_moving else 'stopped'}, travelled {self.conveyor.s:.2f} m"),
            ("Belt slot sensors", self.slot_sensors.text),
            ("E-stop", f"GUI {'PRESSED' if s.gui_estop else 'released'}  |  hardware {self.hw.status}"),
            ("Light curtains", curt),
            ("Area scanner", "OCCUPIED" if p.in_load_zone() else "clear"),
            ("Person", p.status),
            ("Speed", f"{int(100 * self.speed_scale())} %  x{self.sim_speed:g}"
                      + ("  (slowed: person near the fence)" if self.speed_scale() < 1 else "")),
            ("Last event", self.message or "-"),
        ]

    def robot_fields(self, st):
        arm = st.arm
        m = arm.manipulability()
        return [("Doing", st.activity),
                ("Tool x, y, z", "{:.3f}, {:.3f}, {:.3f} m".format(*arm.tcp())),
                ("Manipulability", f"{m:.4f}"),
                ("Obstacle", "IN PATH" if st.obstacle_active else "none"),
                ("Held for obstacle", "YES" if st.blocked else "no"),
                ("Re-plans", str(st.replans))] + st.info()
