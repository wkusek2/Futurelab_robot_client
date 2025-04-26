import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def mark_camera_direction(ax, position, direction, color='red', arrow_scale=1.0):
    """
    Oznacza kierunek patrzenia kamery za pomocą wektora kierunkowego
    
    Parameters:
    -----------
    ax : matplotlib 3D axes
        Osie, na których rysujemy
    position : array-like
        Pozycja kamery [x, y, z]
    direction : array-like
        Wektor kierunku patrzenia [x, y, z]
    color : str
        Kolor oznaczenia
    arrow_scale : float
        Skala wielkości oznaczenia
    """
    # Konwersja do numpy array
    position = np.array(position)
    direction_norm = np.array(direction) / np.linalg.norm(direction)
    
    # Długość strzałki
    length = 3.0 * arrow_scale
    
    # Narysuj linię kierunku
    end_point = position + direction_norm * length
    ax.plot([position[0], end_point[0]], 
            [position[1], end_point[1]], 
            [position[2], end_point[2]], 
            color=color, linewidth=2)
    
    # Dodaj większy marker na końcu linii
    ax.scatter(end_point[0], end_point[1], end_point[2], 
               color=color, marker='^', s=100*arrow_scale)
    
    # Dodaj grot strzałki
    cross_size = length * 0.15
    
    # Znajdź dwa wektory prostopadłe do kierunku
    if abs(direction_norm[0]) < abs(direction_norm[1]):
        perp1 = np.array([1, 0, 0])
    else:
        perp1 = np.array([0, 1, 0])
        
    perp1 = perp1 - direction_norm * np.dot(perp1, direction_norm)
    perp1 = perp1 / np.linalg.norm(perp1) * cross_size
    
    perp2 = np.cross(direction_norm, perp1)
    perp2 = perp2 / np.linalg.norm(perp2) * cross_size
    
    # Narysuj małe linie tworzące "strzałkę"
    for p1, p2 in [(perp1, perp2), (-perp1, perp2), (perp1, -perp2), (-perp1, -perp2)]:
        arrow_point = end_point + p1 + p2 - direction_norm * cross_size * 2
        ax.plot([end_point[0], arrow_point[0]], 
                [end_point[1], arrow_point[1]], 
                [end_point[2], arrow_point[2]], 
                color=color, linewidth=1, alpha=0.5)

def get_direction_from_angle(angle_deg, elevation_deg=0):
    """
    Oblicza wektor kierunku patrzenia kamery na podstawie kąta horyzontalnego
    
    Parameters:
    -----------
    angle_deg : float
        Kąt horyzontalny w stopniach (0 = patrzenie w kierunku +X)
    elevation_deg : float
        Kąt elewacji w stopniach (0 = patrzenie horyzontalnie)
    
    Returns:
    --------
    np.array
        Znormalizowany wektor kierunku [x, y, z]
    """
    angle_rad = np.radians(angle_deg)
    elevation_rad = np.radians(elevation_deg)
    
    # Obliczenie kierunku w układzie sferycznym
    x = np.cos(elevation_rad) * np.cos(angle_rad)
    y = np.cos(elevation_rad) * np.sin(angle_rad)
    z = np.sin(elevation_rad)
    
    return np.array([x, y, z])

def visualize_camera_with_angle(position, angle_deg, elevation_deg=0, color='blue', add_frustum=True, 
                              fov=60, near=0.1, far=5, ax=None, label=None):
    """
    Wizualizacja kamery z określonym kątem patrzenia
    
    Parameters:
    -----------
    position : array-like
        Pozycja kamery [x, y, z]
    angle_deg : float
        Kąt horyzontalny patrzenia kamery w stopniach (0 = patrzenie w kierunku +X)
    elevation_deg : float
        Kąt elewacji kamery w stopniach (0 = patrzenie horyzontalnie)
    color : str
        Kolor wizualizacji
    add_frustum : bool
        Czy dodać wizualizację frustum (pole widzenia) kamery
    fov : float
        Kąt pola widzenia w stopniach
    near : float
        Odległość bliskiej płaszczyzny
    far : float
        Odległość dalekiej płaszczyzny
    ax : matplotlib 3D axes
        Osie, na których rysujemy (jeśli None, tworzone są nowe)
    label : str
        Etykieta kamery
    
    Returns:
    --------
    tuple
        (fig, ax) - figura i osie matplotlib
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = plt.gcf()
    
    # Konwersja do numpy array
    position = np.array(position)
    
    # Obliczenie wektora kierunku
    direction = get_direction_from_angle(angle_deg, elevation_deg)
    
    # Obliczenie punktu docelowego (dla debugowania)
    target = position + direction * far
    
    # Narysuj kamerę jako punkt
    ax.scatter(position[0], position[1], position[2], color=color, s=100, 
              label=label if label else f'Kamera ({angle_deg}°, {elevation_deg}°)')
    
    # Oznacz kierunek patrzenia
    mark_camera_direction(ax, position, direction, color=color, arrow_scale=1.0)
    
    # Dodanie frustum kamery (opcjonalnie)
    if add_frustum:
        # Wektor "w górę" - domyślnie pionowo
        up_vector = np.array([0, 0, 1])
        
        # Jeśli patrzymy prosto w górę lub w dół, musimy dostosować wektor 'up'
        if abs(elevation_deg) > 80:
            up_vector = np.array([0, 1, 0])
        
        # Obliczenie wektorów bazowych kamery
        forward = direction
        right = np.cross(forward, up_vector)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Obliczanie kątów frustum
        half_fov_rad = np.radians(fov / 2)
        aspect_ratio = 1.33  # Proporcje 4:3
        
        # Szerokość i wysokość przy płaszczyźnie bliskiej i dalekiej
        near_height = 2 * np.tan(half_fov_rad) * near
        near_width = near_height * aspect_ratio
        
        far_height = 2 * np.tan(half_fov_rad) * far
        far_width = far_height * aspect_ratio
        
        # Punkty narożników frustum
        # Bliska płaszczyzna
        near_top_left = position + forward * near - right * (near_width/2) + up * (near_height/2)
        near_top_right = position + forward * near + right * (near_width/2) + up * (near_height/2)
        near_bottom_right = position + forward * near + right * (near_width/2) - up * (near_height/2)
        near_bottom_left = position + forward * near - right * (near_width/2) - up * (near_height/2)
        
        # Daleka płaszczyzna
        far_top_left = position + forward * far - right * (far_width/2) + up * (far_height/2)
        far_top_right = position + forward * far + right * (far_width/2) + up * (far_height/2)
        far_bottom_right = position + forward * far + right * (far_width/2) - up * (far_height/2)
        far_bottom_left = position + forward * far - right * (far_width/2) - up * (far_height/2)
        
        # Funkcja pomocnicza do rysowania linii
        def draw_line(p1, p2):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                    color=color, alpha=0.3, linestyle='--')
        
        # Łączenie narożników bliskiej płaszczyzny
        draw_line(near_top_left, near_top_right)
        draw_line(near_top_right, near_bottom_right)
        draw_line(near_bottom_right, near_bottom_left)
        draw_line(near_bottom_left, near_top_left)
        
        # Łączenie narożników dalekiej płaszczyzny
        draw_line(far_top_left, far_top_right)
        draw_line(far_top_right, far_bottom_right)
        draw_line(far_bottom_right, far_bottom_left)
        draw_line(far_bottom_left, far_top_left)
        
        # Łączenie narożników bliskiej i dalekiej płaszczyzny
        draw_line(near_top_left, far_top_left)
        draw_line(near_top_right, far_top_right)
        draw_line(near_bottom_right, far_bottom_right)
        draw_line(near_bottom_left, far_bottom_left)
    
    return fig, ax, [position, target, near_top_left, near_top_right, near_bottom_right, near_bottom_left,
                     far_top_left, far_top_right, far_bottom_right, far_bottom_left]

def visualize_multiple_cameras_with_angles(positions, angles, elevations=None, colors=None, labels=None, 
                                         add_frustum=True, fov=60, near=0.1, far=5, figsize=(12, 10)):
    """
    Wizualizacja wielu kamer z określonymi kątami patrzenia
    
    Parameters:
    -----------
    positions : list of array-like
        Lista pozycji kamer [[x1, y1, z1], [x2, y2, z2], ...]
    angles : list of float
        Lista kątów horyzontalnych patrzenia kamer w stopniach
    elevations : list of float, optional
        Lista kątów elewacji patrzenia kamer w stopniach
    colors : list of str, optional
        Lista kolorów dla kamer
    labels : list of str, optional
        Lista etykiet dla kamer
    add_frustum : bool, optional
        Czy dodać wizualizację frustum (pole widzenia) kamery
    fov : float, optional
        Kąt pola widzenia w stopniach
    near : float, optional
        Odległość bliskiej płaszczyzny
    far : float, optional
        Odległość dalekiej płaszczyzny
    figsize : tuple, optional
        Rozmiar wykresu
    
    Returns:
    --------
    tuple
        (fig, ax) - figura i osie matplotlib
    """
    if colors is None:
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta']
    
    if labels is None:
        labels = [f'Kamera {i+1}' for i in range(len(positions))]
        
    if elevations is None:
        elevations = [0] * len(positions)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    all_points = []
    
    for i, (position, angle, elevation) in enumerate(zip(positions, angles, elevations)):
        color = colors[i % len(colors)]
        label = labels[i]
        
        # Dodanie kamery
        _, _, points = visualize_camera_with_angle(
            position, angle, elevation, color=color, 
            add_frustum=add_frustum, fov=fov, near=near, far=far, 
            ax=ax, label=label
        )
        all_points.extend(points)
    
    # Automatyczne ustawienie granic wykresu
    all_points = np.array([p for p in all_points if p is not None])
    
    if len(all_points) > 0:
        x_min, y_min, z_min = np.min(all_points, axis=0) - 1
        x_max, y_max, z_max = np.max(all_points, axis=0) + 1
        
        # Ustal równe osie
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
        mid_x = (x_max + x_min) / 2
        mid_y = (y_max + y_min) / 2
        mid_z = (z_max + z_min) / 2
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Ustawienia wykresu
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Wizualizacja kamer z określonymi kątami')
    ax.legend()
    
    plt.tight_layout()
    return fig, ax

# Przykład użycia - wizualizacja kamer z różnymi kątami
if __name__ == "__main__":
    # Definiujemy pozycje kamer i kąty patrzenia
    camera_positions = [
        [0, 0, 0],           # Kamera 1
        [5, 0, 2],           # Kamera 2
        [2.5, 5, 1]          # Kamera 3
    ]
    
    # Kąty patrzenia (w stopniach, 0 = kierunek +X, 90 = kierunek +Y)
    camera_angles = [30, 150, 270]
    
    # Kąty elewacji (w stopniach, 0 = poziomo, 90 = pionowo w górę)
    camera_elevations = [0, 15, -10]
    
    # Etykiety kamer
    labels = ["Kamera główna (30°)", "Kamera boczna (150°)", "Kamera górna (270°)"]
    
    # Kolory kamer
    colors = ["blue", "red", "green"]
    
    # Wizualizacja kamer
    fig, ax = visualize_multiple_cameras_with_angles(
        camera_positions, 
        camera_angles,
        elevations=camera_elevations,
        colors=colors, 
        labels=labels, 
        add_frustum=True,
        fov=60,             # Kąt pola widzenia
        near=0.5,           # Bliska płaszczyzna
        far=7               # Daleka płaszczyzna
    )
    
    # Dodanie obiektu w scenie
    scene_object = [2.5, 2.5, 1]
    ax.scatter(scene_object[0], scene_object[1], scene_object[2], 
              color='black', marker='*', s=200, label='Obiekt sceny')
    
    # Dodanie siatki w płaszczyźnie XY dla lepszej orientacji
    x_grid = np.linspace(-5, 10, 4)
    y_grid = np.linspace(-5, 10, 4)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    
    ax.plot_wireframe(X, Y, Z, color='gray', alpha=0.2)
    
    plt.show()