
import cv2
import numpy as np

# Metoda do dodania do klasy Camera

def    convert_camera_to_global(R_camera, T_camera, R_ref, T_ref):
        """
        Przekształca pozycję i orientację kamery do układu globalnego.
        
        Args:
            R_camera (ndarray): Macierz rotacji kamery do przekształcenia (3x3).
            T_camera (ndarray): Wektor translacji kamery do przekształcenia (3x1 lub 3,).
            R_ref (ndarray): Macierz rotacji układu odniesienia (3x3).
            T_ref (ndarray): Wektor translacji układu odniesienia (3x1 lub 3,).
            
        Returns:
            tuple: (R_global, T_global) - Przekształcona macierz rotacji i wektor translacji.
        """
        # Upewnij się, że wektory translacji mają odpowiedni format
        if T_camera.ndim == 1:
            T_camera = T_camera.reshape(-1, 1)
        if T_ref.ndim == 1:
            T_ref = T_ref.reshape(-1, 1)
        
        # Oblicz globalną rotację i translację
        R_global = R_ref @ R_camera
        T_global = R_ref @ T_camera + T_ref
        
        return R_global, T_global

def triangulate_point(point_cam1, point_cam2, K1, R1, T1, K2, R2, T2):
    """
    Trianguluje punkt 3D na podstawie punktów 2D z dwóch kamer.
    
    Args:
        point_cam1 (array-like): Punkt 2D z pierwszej kamery [x, y]
        point_cam2 (array-like): Punkt 2D z drugiej kamery [x, y]
        K1, K2: Macierze wewnętrzne kamer
        R1, R2: Macierze rotacji kamer
        T1, T2: Wektory translacji kamer
        
    Returns:
        np.ndarray: Punkt 3D [x, y, z] w układzie globalnym
    """
    # Upewnij się, że wektory translacji mają odpowiedni format
    
    # Utwórz macierze projekcji
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Kamera 1: [R | T] = [I | 0]
    P1 = K1 @ P1

    P2 = np.hstack((R1, T1.reshape(-1, 1)))  # Kamera 2: [R | T]
    P2 = K2 @ P2
    
    # Przekształć punkty do formatu wymaganego przez cv2.triangulatePoints
    points1 = np.array([point_cam1], dtype=np.float32).reshape(2, 1)
    points2 = np.array([point_cam2], dtype=np.float32).reshape(2, 1)
    
    # Triangulacja punktu
    points4D = cv2.triangulatePoints(P1, P2, points1, points2)
    
    # Konwersja punktów z jednorodnych (homogeneous) na zwykłe współrzędne 3D
    point3D = (points4D[:3] / points4D[3]).T
    
    return point3D[0]  # Zwróć punkt jako wektor [x, y, z]

def triangulate_with_marker_reference(point_cam1, point_cam2, K1, K2, stereo_R, stereo_T, marker_R, marker_T):
    """
    Trianguluje punkt 3D z punktów 2D z uwzględnieniem układu odniesienia markera.
    
    Args:
        point_cam1 (array-like): Punkt 2D z pierwszej kamery [x, y]
        point_cam2 (array-like): Punkt 2D z drugiej kamery [x, y]
        K1, K2: Macierze wewnętrzne kamer
        stereo_R: Macierz rotacji z kalibracji stereo (kamera2 względem kamery1)
        stereo_T: Wektor translacji z kalibracji stereo
        marker_R: Macierz rotacji markera względem kamery1
        marker_T: Wektor translacji markera względem kamery1
        
    Returns:
        np.ndarray: Punkt 3D w układzie globalnym (markera)
    """
    # Macierz rotacji i translacji dla kamery 1 w układzie markera
    R1 = marker_R
    T1 = marker_T
    
    # Oblicz macierz rotacji i translacji dla kamery 2 w układzie markera
    R2, T2 = convert_camera_to_global(stereo_R, stereo_T, marker_R, marker_T)
    
    # Triangulacja w układzie markera
    return triangulate_point(point_cam1, point_cam2, K1, R1, T1, K2, R2, T2)

class Triangulation:
    def __init__(self):
        self.frame = None

        self.obj0 = None
        self.obj1 = None

        self.K1 = np.array([
                [1.76665904e+03, 0.00000000e+00, 6.02400704e+02],
                [0.00000000e+00, 1.76930355e+03, 1.12010051e+03],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
            ])

        self.K2 = np.array([
                [1.77465827e+03, 0.00000000e+00, 6.06988235e+02],
                [0.00000000e+00, 1.76724437e+03, 1.18724507e+03],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
            ])

            # Macierz rotacji i translacji dla układu stereo
        self.R_stereo = np.array([
                [0.995573, 0.003629, -0.093926],
                [-0.048177, 0.877720, -0.476746],
                [0.080710, 0.479160, 0.874009]
            ])

        self.T_stereo = np.array([0.068145, 0.124145, 0.154153])

    def object_is_detected(self, obj0):
        return obj0 is not None and len(obj0) > 0
         

    def get_3d_position(self, obj0, obj1):
            # Pobierz współrzędne wykrytych obiektów
            if not self.object_is_detected(obj0) or not self.object_is_detected(obj1):
                return None

            # Pobierz informacje o wykrytych obiektach
            self.obj0 = obj0
            self.obj1 = obj1

            # Oblicz środki obiektów
            center0 = (( (self.obj0[1] + self.obj0[3]) / 2), (self.obj0[0] + self.obj0[2]) / 2)
            center1 = ((self.obj1[1] + self.obj1[3]) / 2,(self.obj1[0] + self.obj1[2]) / 2)


            # Utwórz bezpośrednio macierze projekcji
            P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Kamera 1: [I | 0]
            P1 = self.K1 @ P1

            P2 = np.hstack((self.R_stereo, self.T_stereo.reshape(-1, 1)))  # Kamera 2: [R | T]
            P2 = self.K2 @ P2

            # Przekształć punkty do formatu wymaganego przez cv2.triangulatePoints
            points1 = np.array(center0, dtype=np.float32).reshape(2, 1)
            points2 = np.array(center1, dtype=np.float32).reshape(2, 1)

            # Triangulacja bezpośrednio przez cv2
            try:
                points4D = cv2.triangulatePoints(P1, P2, points1, points2)

                # Konwersja na zwykłe współrzędne 3D
                point3D = (points4D[:3] / points4D[3]).reshape(3)

                # print(f"Raw point3D: {point3D}")

                # Sprawdź czy odległość jest realistyczna
                distance = np.linalg.norm(point3D)
                # print(f"Raw distance: {distance} m, {distance*100} cm")

                # Wyreguluj skalę na podstawie znanej odległości (jeśli masz punkt odniesienia)
                # Przykład: jeśli wiesz, że obiekt jest na 50 cm
                # scale_factor = 50.0 / (distance * 100)
                # point3D *= scale_factor

                return point3D

            except Exception as e:
                print(f"Błąd triangulacji: {e}")
                return None

