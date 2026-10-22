import numpy as np


class RobotStation:
    number = 0  # robot 1 .. 4 along the line
    key = ""  # short name used on the terminal ("final <key> d1 .. d6")
    name = ""  # arm name ("MH5", "MPX3500", "FANUC", "UR3e")
    title = ""  # what the station does
    model = ""  # robot make / model
    owner = ""  # group member who wrote this robot's code

    arm = None  # shared.motion.Arm (or a subclass)
    home_q = None  # pose the robot returns to between jobs and before production starts
    final_q = None  # pose it goes to at the end of production (can be changed from the pendant / terminal)
    sensor = None  # the sensor that tells this robot a car is in front of it (or None)

    def run(self, line):
        """This robot's part of production (generator)."""
        return
        yield  # noqa: unreachable - makes this a generator

    def draw(self):
        pass

    def set_obstacle(self, on):
        pass

    @property
    def obstacle_active(self):
        return bool(self.arm.obstacle.active)

    @property
    def blocked(self):
        return bool(self.arm.blocked)

    @property
    def replans(self):
        return self.arm.replans

    @property
    def activity(self):
        """One line saying what the robot is doing right now (the pendant's robot list)."""
        return self.arm.note

    def info(self):
        return []

    def reset(self):
        pass

    def at_home(self, tol=np.deg2rad(0.5)):
        return bool(np.all(np.abs(self.arm.q - np.asarray(self.home_q, float)) < tol))
