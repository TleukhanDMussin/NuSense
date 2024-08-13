import cv2
import numpy as np

# Define input and output file paths
image_path = r"D:/Experiments/2/Photos_Fisheyed/x1_z2.jpg"
image_out_path = r"D:/Experiments/2/Photos_Defisheyed/x1_z2.jpg"

# Load the fisheye image
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
    exit()

# Get the width and height of the image
h, w = image.shape[:2]

# Define the camera matrix K and distortion coefficients D
K = np.array([[w, 0, w // 2], [0, w, h // 2], [0, 0, 1]], dtype=np.float64)
D = np.array([-0.28, 0.07, 0.0, 0.0], dtype=np.float64)

# Estimate new camera matrix for undistort rectify
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=1.0)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)

# Undistort the image
undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# Save the undistorted image
cv2.imwrite(image_out_path, undistorted_image)

# Display the undistorted image
cv2.imshow('Undistorted Image', undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Undistorted image saved to: {image_out_path}")
