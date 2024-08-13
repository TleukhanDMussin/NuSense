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

    # Sort control points by y first to form rows
    control_points.sort(key=lambda k: k[1])

    # Determine grid size (rows and columns)
    num_points = len(control_points)
    rows = 11  # Fixed number of rows
    cols = (num_points + rows - 1) // rows  # Ceiling division to handle non-square grids

    # Ensure control points form a complete grid by padding if necessary
    grid_ctrlpts = []
    for i in range(0, num_points, cols):
        row = control_points[i:i + cols]
        row.sort(key=lambda k: k[0])  # Sort each row by x
        if len(row) < cols:
            # Pad the row with dummy points if it's not full
            row.extend([[0, 0, 0.0]] * (cols - len(row)))
        grid_ctrlpts.append(row)

    # Create a BSpline surface
    surf = BSpline.Surface()

    # Set degrees
    surf.degree_u = 3
    surf.degree_v = 3

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

def classify_shape(contour, min_area=50):
    # Compute the area of the contour
    area = cv2.contourArea(contour)
    
    # Restrict shapes with area less than min_area
    if area < min_area:
        return None
    
    # Compute the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h

    # Compute the extent
    rect_area = w * h
    extent = float(area) / rect_area

    # Compute the perimeter and the perimeter to area ratio
    perimeter = cv2.arcLength(contour, True)
    perimeter_to_area_ratio = perimeter / area

    # Use geometric properties to classify the shape
    if area < 100:  # Adjust threshold based on your application
        return "pin"
    elif aspect_ratio > 1.7 or aspect_ratio < 0.8:  # Line-like shapes
        return "edge"
    elif extent < 0.5:  # Check if the extent is low (indicating a line)
        return "edge"
    else:
        return "pin"

def calculate_angle(contour):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        # Angle calculation
        angle = 0.5 * np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"])
        angle_deg = np.degrees(angle)
        return angle_deg
    return None

# Process both images
P1, control_points_1 = process_image(r'C:/Users/Tleukhan/Pictures/Camera Roll/Defisheye_Images/No_pressure_output.jpg')
P2, control_points_2 = process_image(r'C:/Users/Tleukhan/Pictures/Camera Roll/Defisheye_Images/Pressure_2_output.jpg')

# Compute the Euclidean distance between P2 and P1
P_diff = np.linalg.norm(np.array(P2) - np.array(P1), axis=1)

# Sum the Euclidean distances
total_distance = np.sum(P_diff)

print(P_diff)

# Print the total Euclidean distance
print(f'Total Euclidean Distance: {total_distance}')

# Create a binary image based on Euclidean distances
binary_image = np.zeros_like(P_diff, dtype=np.uint8)
binary_image[P_diff > 11] = 255

# Extract x and y coordinates for plotting
x_coords_1 = [pt[0] for pt in P1]
y_coords_1 = [pt[1] for pt in P1]

# Create an image to visualize the binary result
height, width = 1000, 1380  # Size of the image
binary_image_visual = np.zeros((height, width), dtype=np.uint8)

# Radius for the circles to make dots larger
radius = 2

for (x, y), value in zip(zip(x_coords_1, y_coords_1), binary_image):
    x = min(max(int(x), 0), width - 1)
    y = min(max(int(y), 0), height - 1)
    if value == 255:
        cv2.circle(binary_image_visual, (x, y), radius, (255, 255, 255), -1)

# Apply dilation to combine close white regions
kernel = np.ones((7, 7), np.uint8)  # You can adjust the kernel size as needed
binary_image_visual = cv2.dilate(binary_image_visual, kernel, iterations=2)

# Apply connected component analysis to keep only the largest component
num_labels, labels_im = cv2.connectedComponents(binary_image_visual)

# Find the largest connected component
largest_component = 1 + np.argmax(np.bincount(labels_im.flat)[1:])

# Create a mask for the largest component
largest_component_mask = (labels_im == largest_component).astype(np.uint8) * 255

# Plot the binary image with the largest component
plt.figure(figsize=(10, 6))
plt.imshow(largest_component_mask, cmap='gray')
plt.title('Binary Image with Largest Connected Component')
plt.xlabel('X')
plt.ylabel('Y')
#plt.gca().invert_yaxis()  # Invert Y axis to match the image coordinate system
plt.grid(True)
plt.savefig('largest_connected_component.png')
plt.show()

# Find contours in the binary image with the largest component
contours, _ = cv2.findContours(largest_component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over the contours to classify and print the shape and angle
for contour in contours:
    shape = classify_shape(contour)
    angle = calculate_angle(contour)
    if shape is not None and angle is not None:
        print(f"Detected shape: {shape}, Angle: {angle:.2f} degrees")

        # Draw the contour and label it on the image
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            cv2.drawContours(largest_component_mask, [contour], -1, (0, 255, 0), 2)
            cv2.putText(largest_component_mask, f"{shape}, {angle:.2f}°", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Display the result
cv2.imshow("Shape Detection with Angles", largest_component_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
