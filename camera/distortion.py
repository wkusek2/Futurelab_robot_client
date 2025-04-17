import numpy as np
import cv2

display_w = 1296
display_h = 2304


K1 = np.array([[1.76665904e+03, 0.00000000e+00, 6.02400704e+02],
                    [0.00000000e+00, 1.76930355e+03, 1.12010051e+03],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

D1 = np.array([[ 0.00279605,  0.36580486, -0.00901127, -0.01083656, -0.56823121]])

K2 = np.array([[1.77465827e+03, 0.00000000e+00, 6.06988235e+02],
                [0.00000000e+00, 1.76724437e+03, 1.18724507e+03],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

D2 = np.array([[ 0.08981312, -0.56558791,  0.00496143, -0.00749746,  1.08375072]])

newcameramtx1, roi1 = cv2.getOptimalNewCameraMatrix(K1, D1, (display_w, display_h), 1, (display_w, display_h))
newcameramtx2, roi2 = cv2.getOptimalNewCameraMatrix(K2, D2, (display_w, display_h), 1, (display_w, display_h))

def distortion(image0, image1):
    # Undistort
    undistorted0 = cv2.undistort(image0, K1, D1, None, newcameramtx1)
    undistorted1 = cv2.undistort(image1, K2, D2, None, newcameramtx2)

    # Crop the image
    x0, y0, w0, h0 = roi1
    x1, y1, w1, h1 = roi2

    undistorted0 = undistorted0[y0:y0 + h0, x0:x0 + w0]
    undistorted1 = undistorted1[y1:y1 + h1, x1:x1 + w1]

    return undistorted0, undistorted1


