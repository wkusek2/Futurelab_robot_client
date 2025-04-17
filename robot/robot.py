import numpy as np

from robot.kinematics import inverse_kinematics
from robot.matrices import *

class Robot:
    def __init__(self) -> None:
        self.rx = 0
        self.ry = 0
        self.rz = 0

    def update_robot(self, x, y, z):
        try:
            theta1, theta2, theta3, theta4, pos1, pos2, pos3, pos4, valid_position = inverse_kinematics(x, y, z)

            if theta1 is not None:
                return theta1, theta2, theta3, theta4, pos1, pos2, pos3, pos4
        except ValueError:
            print("ValueError")

        return 0, 0, 0, 0, 0, 0, 0, 0,

    def compute_end_pos(self, theta1, theta2, theta3, theta4, a3, a4, a5):
        self.rx, self.ry, self.rz = compute_end_pos(theta1, theta2, theta3, theta4, a3, a4, a5)

    def t_ends(self, theta1, theta2, theta3, theta4, a3, a4, a5):
        T1_end = T1(theta1)
        T2_end = np.dot(T1_end, T2(theta2))
        T3_end = np.dot(T2_end, T3(theta3, a3))
        T4_end = np.dot(T3_end, T4(theta4, a4))
        T5_end = np.dot(T4_end, T5(a5))

        points = np.array([
            [0, 0, 0],
            T1_end[:3, 3],
            T2_end[:3, 3],
            T3_end[:3, 3],
            T4_end[:3, 3],
            T5_end[:3, 3]
        ])

        return points
