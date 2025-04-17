import math
import numpy as np
import matplotlib.pyplot as plt

from robot.matrices import *

a3 = 152.794
a4 = 157.76
a5 = 90

class Plot:
    def __init__(self) -> None:

        plt.switch_backend('agg')
        self.fig = plt.figure(figsize=(10, 8), dpi=70)
        self.ax = self.fig.add_subplot(111, projection='3d')

    def plot_robot(self, robot, theta1, theta2, theta3, theta4):
        
        self.ax.plot([0], [0], [0], 'go', label='Base')

        robot.compute_end_pos(theta1, theta2, theta3, theta4, a3, a4, a5)
        points = robot.t_ends(theta1, theta2, theta3, theta4, a3, a4, a5)

        self.ax.plot(points[:, 0], points[:, 1], points[:, 2], 'bo-', label='Links')
        self.ax.plot(robot.rx, robot.ry, robot.rz, 'ro', label='End-Effector')

        self.ax.set_xlim(-300, 300)
        self.ax.set_ylim(-300, 300)
        self.ax.set_zlim(-30, 400)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        plt.gca().invert_zaxis()

        self.ax.set_title('3D Robot Arm')
        self.ax.legend()

    def plot_camera(self, x, y, z):
        cord = [x, y, z]
        w_p = [121, -30, 70]

        # Rysujemy punkt i zapisujemy referencję do niego
        camera_point = self.ax.scatter(
            (w_p[0]+cord[0]*1000),
            (w_p[1]+cord[1]*1000),
            (w_p[2]+cord[2]*1000), 
            c=['blue'], 
            s=[100]
        )

        # Zwracamy referencję do punktu
        return camera_point
        

