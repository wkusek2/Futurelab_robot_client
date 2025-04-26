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
        self.ax.cla()
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
    
    def camera_vis(self, angle_deg=0, elevation_deg=0, scale=10):
        """
        Zoptymalizowana funkcja rysująca kamerę z określonym kątem oraz frustum
        """
        # Pozycja kamery
        camera_pos = np.array([50, 50, 50])

        # Obliczenie wektora kierunku
        angle_rad = np.radians(angle_deg)
        elevation_rad = np.radians(elevation_deg)

        # Kierunek jako wektor jednostkowy
        direction = np.array([
            np.cos(elevation_rad) * np.cos(angle_rad),
            np.cos(elevation_rad) * np.sin(angle_rad),
            np.sin(elevation_rad)
        ])

        # Optymalizacja - używamy wektorów zamiast indywidualnych zmiennych
        # Obliczenie wektorów bazowych kamery
        up_vector = np.array([0, 0, 1]) if abs(elevation_deg) <= 80 else np.array([0, 1, 0])

        right = np.cross(direction, up_vector)
        right = right / np.linalg.norm(right)

        up = np.cross(right, direction)

        # Parametry frustum
        fov = 60  # Kąt pola widzenia w stopniach
        aspect_ratio = 1.33  # Proporcje 4:3
        near = 5 * scale
        far = 20 * scale

        # Kąt i wymiary frustum
        half_fov_rad = np.radians(fov / 2)
        near_height = 2 * np.tan(half_fov_rad) * near
        near_width = near_height * aspect_ratio
        far_height = 2 * np.tan(half_fov_rad) * far
        far_width = far_height * aspect_ratio

        # Obliczenie punktów frustum w jednym kroku za pomocą wektorów
        # Bliska płaszczyzna
        near_points = np.array([
            camera_pos + direction * near - right * (near_width/2) + up * (near_height/2),  # top_left
            camera_pos + direction * near + right * (near_width/2) + up * (near_height/2),  # top_right
            camera_pos + direction * near + right * (near_width/2) - up * (near_height/2),  # bottom_right
            camera_pos + direction * near - right * (near_width/2) - up * (near_height/2)   # bottom_left
        ])

        # Daleka płaszczyzna
        far_points = np.array([
            camera_pos + direction * far - right * (far_width/2) + up * (far_height/2),    # top_left
            camera_pos + direction * far + right * (far_width/2) + up * (far_height/2),    # top_right
            camera_pos + direction * far + right * (far_width/2) - up * (far_height/2),    # bottom_right
            camera_pos + direction * far - right * (far_width/2) - up * (far_height/2)     # bottom_left
        ])

        # Punkt końcowy wektora kierunku
        end_point = camera_pos + direction * 30 * scale

        # Używamy jednej kolekcji dla wszystkich linii zamiast wielu wywołań plot()
        lines = []

        # Linia kierunku
        lines.append((camera_pos, end_point))

        # Krawędzie bliskiej płaszczyzny
        for i in range(4):
            lines.append((near_points[i], near_points[(i+1)%4]))

        # Krawędzie dalekiej płaszczyzny
        for i in range(4):
            lines.append((far_points[i], far_points[(i+1)%4]))

        # Połączenia między płaszczyznami
        for i in range(4):
            lines.append((near_points[i], far_points[i]))

        # Konwersja do formatu dla Line3DCollection
        line_segments = []
        for p1, p2 in lines:
            line_segments.append([p1, p2])

        # Kolory dla różnych segmentów
        colors = ['r'] + ['b']*4 + ['g']*4 + ['y']*4

        # Tworzenie jednej kolekcji linii zamiast wielu wywołań plot()
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        lc = Line3DCollection(line_segments, colors=colors, linewidths=2, linestyles='solid')
        line_collection = self.ax.add_collection3d(lc)

        # Dodanie punktu kamery
        camera_point = self.ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], 
                                      color='red', s=100, label='Kamera')

        # Zwracamy kolekcję i punkt - mniej obiektów do zarządzania
        return [line_collection, camera_point]



