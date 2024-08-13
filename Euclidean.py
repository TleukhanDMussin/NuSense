import cv2
import numpy as np
import matplotlib.pyplot as plt
from geomdl import BSpline

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for blue color in HSV
    lower_blue = np.array([115, 160, 160])
    upper_blue = np.array([140, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours in the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to store the coordinates of the control points
    control_points = []

    # Loop over the contours to compute the centroid of each contour
    for contour in contours:
        # Compute the moments of the contour
        M = cv2.moments(contour)
        if M['m00'] != 0:
            # Compute the x, y coordinates of the centroid
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            control_points.append([cX, cY, 0.0])  # Add z-coordinate as 0.0
        else:
            # Handle the case where the contour area is zero
            x, y, w, h = cv2.boundingRect(contour)
            cX = int(x + w / 2)
            cY = int(y + h / 2)
            control_points.append([cX, cY, 0.0])  # Add z-coordinate as 0.0

    control_points.sort(key=lambda k: k[1])

    # Determine grid size (rows and columns)
    num_points = len(control_points)
    rows = 11  # Fixed number of rows
    cols = (num_points + rows - 1) // rows  # Ceiling division to handle non-square grids

    grid_ctrlpts = []
    for i in range(0, num_points, cols):
        row = control_points[i:i + cols]
        row.sort(key=lambda k: k[0])  # Sort each row by x
        grid_ctrlpts.append(row)

    # Create a BSpline surface
    surf = BSpline.Surface()

    # Set degrees
    surf.degree_u = 1
    surf.degree_v = 1

    # Set control points
    surf.ctrlpts2d = grid_ctrlpts

    # Calculate the knot vectors
    def generate_knot_vector(degree, num_ctrlpts):
        num_knots = num_ctrlpts + degree + 1
        knots = np.zeros(num_knots)
        knots[degree:num_knots-degree] = np.linspace(0, 1, num_knots - 2*degree)
        knots[num_knots-degree:] = 1
        return knots.tolist()

    surf.knotvector_u = generate_knot_vector(surf.degree_u, len(grid_ctrlpts))
    surf.knotvector_v = generate_knot_vector(surf.degree_v, len(grid_ctrlpts[0]))

    # Set evaluation delta
    surf.delta = 0.015

    # Evaluate surface points
    surf.evaluate()

    # Retrieve the evaluated points (P(u,v) values)
    evaluated_points = surf.evalpts

    return evaluated_points, control_points

# Process both images
P1, control_points_1 = process_image(r'D:/Experiments/2/Photos_Output/reference.jpg')
P2, control_points_2 = process_image(r'D:/Experiments/2/Photos_Output/x1_z2.jpg')

# Compute the Euclidean distance between P2 and P1
P_diff = np.linalg.norm(np.array(P2) - np.array(P1), axis=1)

# Sum the Euclidean distances
total_distance = np.sum(P_diff)

# Print the Euclidean distances
print(P_diff)

# Print the total Euclidean distance
print(f'Total Euclidean Distance: {total_distance}')

# Filter points with Euclidean distances less than 10
filtered_indices = P_diff >= 0
filtered_P_diff = P_diff[filtered_indices]

filtered_P1 = np.array(P1)[filtered_indices]
filtered_P2 = np.array(P2)[filtered_indices]

# Extract x and y coordinates for plotting
x_coords_1 = [pt[0] for pt in P1]
y_coords_1 = [pt[1] for pt in P1]

x_coords_1_filtered = [pt[0] for pt in filtered_P1]
y_coords_1_filtered = [pt[1] for pt in filtered_P1]

# Plot the Euclidean distances as points with varying size
plt.figure(figsize=(10, 6))
plt.scatter(x_coords_1_filtered, y_coords_1_filtered, s=filtered_P_diff * 5, c=filtered_P_diff, cmap='viridis', alpha=0.7, edgecolors='w', linewidth=0.5)
plt.colorbar(label='Euclidean Distance')
plt.title('Euclidean Distances Between Evaluated Points (P2 - P1)')
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().invert_yaxis()  # Invert Y axis to match the image coordinate system
plt.grid(True)
plt.savefig('euclidean_distances.png')
plt.show()

# 2D Scatter plot of the control points for the first image
plt.figure(figsize=(10, 6))
plt.scatter(x_coords_1, y_coords_1, c='blue', marker='o')
plt.title('Control Points (XY Plane) - Image 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().invert_yaxis()  # Invert Y axis to match the image coordinate system
plt.grid(True)
plt.savefig('control_points_image1.png')
plt.show()

# 2D Scatter plot of the control points for the second image
plt.figure(figsize=(10, 6))
plt.scatter([pt[0] for pt in P2], [pt[1] for pt in P2], c='green', marker='o')
plt.title('Control Points (XY Plane) - Image 2')
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().invert_yaxis()  # Invert Y axis to match the image coordinate system
plt.grid(True)
plt.savefig('control_points_image2.png')
plt.show()
