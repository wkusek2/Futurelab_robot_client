import numpy as np

# Define the transformation matrices
def T1(theta1, d1=0):
    return np.array([
        [np.cos(theta1), -np.sin(theta1), 0, 0],
        [np.sin(theta1), np.cos(theta1), 0, 0],
        [0, 0, 1, d1],
        [0, 0, 0, 1]
    ])
def T2(theta2):
    return np.array([
        [np.cos(theta2), -np.sin(theta2), 0, 0],
        [0, 0, 1, 0],
        [-np.sin(theta2), -np.cos(theta2), 0, 0],
        [0, 0, 0, 1]
    ])
def T3(theta3, a3):
    return np.array([
        [np.cos(theta3), -np.sin(theta3), 0, a3],
        [np.sin(theta3), np.cos(theta3), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
def T4(theta4, a4):
    return np.array([
        [np.cos(theta4), -np.sin(theta4), 0, a4],
        [np.sin(theta4), np.cos(theta4), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
def T5(a5):
    return np.array([
        [1, 0, 0, a5],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    

def compute_end_pos(theta1, theta2, theta3, theta4, a3=152.794, a4=157.76, a5=90):
    T1_matrix = T1(theta1)
    T2_matrix = np.dot(T1_matrix, T2(theta2))
    T3_matrix = np.dot(T2_matrix, T3(theta3, a3))
    T4_matrix = np.dot(T3_matrix, T4(theta4, a4))
    T5_matrix = np.dot(T4_matrix, T5(a5))
    
    z1 = T1_matrix[2, 3]
    z2 = T2_matrix[2, 3]
    z3 = T3_matrix[2, 3]
    z4 = T4_matrix[2, 3]
    z5 = T5_matrix[2, 3]
    
    if  z3 <= 0 or z4 <= 0 or z5 <= 0:
        raise ValueError("Jeden z punktów znajduje się poniżej lub na poziomie 0 w osi Z")
    
    return T5_matrix[0, 3], T5_matrix[1, 3], z5
