# This is the code to process the NUSense
import cv2
import numpy as np
from geomdl import BSpline
import matplotlib.pyplot as plt

def defisheye(img):
    """Convert fisheye image to undistorted.
    Input:
        - img (cv2 format): input image 
        - img_undistorted (cv2_format): output image
    """
    # Get the width and height of the image
    h, w = img.shape[:2]

    print(h,w)

    # Define the camera matrix K and distortion coefficients D
    K = np.array([[w, 0, w // 2], [0, w, h // 2], [0, 0, 1]], dtype=np.float64)
    D = np.array([-0.28, 0.07, 0.0, 0.0], dtype=np.float64)

    # Estimate new camera matrix for undistort rectify
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=1.0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)

    # Undistort the image
    img_undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return img_undistorted


def visiualize(img, window_name="Image"):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class NUSense:
    def __init__(self):
        pass

    def apply_sobel_with_bilateral_filter_to_image(self, img, roi_x, roi_y, roi_w, roi_h, distance_threshold=50):
        """ Preprocessing via filtering
        input:
            - img (CV2 format): input image
            - roi_x (int): 400
            - roi_y (int): 203
            - roi_w (int): 1290
            - roi_h (int): 990
            - distance_threshold (int, default=50):50
        output:
            - img_roi (CV2 format): filtered image
        """

        # Apply ROI
        img_roi = img[roi_y:roi_y + roi_h, roi_x: roi_x + roi_w]
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Bilateral Filter
        bilateral_filtered = cv2.bilateralFilter(gray, 20, 65, 75)
        
        # Apply Sobel filter
        sobelx = cv2.Sobel(bilateral_filtered, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(bilateral_filtered, cv2.CV_64F, 0, 1, ksize=3)

        abs_sobelx = cv2.convertScaleAbs(sobelx)
        abs_sobely = cv2.convertScaleAbs(sobely)
        
        # Combine the two images
        sobel_combined = cv2.addWeighted(abs_sobelx, 0.86, abs_sobely, 0.86, 0)
        
        # Threshold the Sobel result to get binary image
        _, binary = cv2.threshold(sobel_combined, 19, 255, cv2.THRESH_BINARY)
        
        # Apply dilation followed by erosion
        kernel_dilation = np.ones((3, 3), np.uint8)
        kernel_erosion = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel_dilation, iterations=1)
        closed_binary = cv2.erode(dilated, kernel_erosion, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(closed_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        quadrilaterals = []
        
        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.08 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                # Filter out too small or too large areas
                if 2000 < area < 100000 and solidity > 0.2:
                    cv2.drawContours(img_roi, [approx], 0, (0, 255, 0), 2)
                    quadrilaterals.append(approx)
        
        # Connect corners of quadrilaterals and highlight the center of the blue lines
        for quad1 in quadrilaterals:
            for point1 in quad1:
                min_dist = float('inf')
                closest_point = None
                for quad2 in quadrilaterals:
                    if np.array_equal(quad1, quad2):
                        continue
                    for point2 in quad2:
                        dist = np.linalg.norm(point1 - point2)
                        if dist < min_dist and dist < distance_threshold:
                            min_dist = dist
                            closest_point = point2
                if closest_point is not None:
                    # Draw the blue line
                    cv2.line(img_roi, tuple(point1[0]), tuple(closest_point[0]), (255, 255, 255), 2)
                    
                    # Calculate the midpoint
                    midpoint = ((point1[0][0] + closest_point[0][0]) // 2, (point1[0][1] + closest_point[0][1]) // 2)
                    
                    # Draw a small circle at the midpoint
                    cv2.circle(img_roi, midpoint, 3, (255, 0, 0), -1)

        return img_roi

    def calculate_euclidean(self, img_ref, img_curr):
        '''
        TODO
        '''
        # Process both images
        P1, control_points_1 = self._evaluates_points(img_ref)
        P2, control_points_2 = self._evaluates_points(img_curr)

        # Compute the Euclidean distance between P2 and P1
        P_diff = np.linalg.norm(np.array(P2) - np.array(P1), axis=1)

        # Sum the Euclidean distances
        total_distance = np.sum(P_diff)

        return P_diff, P1, P2

    def _evaluates_points(self, img):
        """
        TODO
        """

        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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

    def visualize_euclidean_distances(self, P1, P2, P_diff):
        """
        TODO
        """
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



if __name__ == '__main__':

    # get undistorted img
    img_path = "data/experiment1/8N/x1_z1.jpg"
    img_ref_path = "data/experiment1/8N/reference.jpg"
    img = cv2.imread(img_path)
    img_ref = cv2.imread(img_path)

    # correct fisheye image
    img_undistorted = defisheye(img)
    img_ref_undistorted = defisheye(img_ref)


    #visiualize(img_undistorted)

    nusense = NUSense()

    # preprocess by filtering
    distance_threshold = 50
    roi_x, roi_y, roi_w, roi_h = 390, 203, 1335, 1000
    img_roi = nusense.apply_sobel_with_bilateral_filter_to_image( img=img_undistorted, 
                                                            roi_x=roi_x, 
                                                            roi_y=roi_y, 
                                                            roi_w=roi_w, 
                                                            roi_h=roi_h, 
                                                            distance_threshold=distance_threshold)
    img_ref_roi = nusense.apply_sobel_with_bilateral_filter_to_image( img=img_ref_undistorted, 
                                                            roi_x=roi_x, 
                                                            roi_y=roi_y, 
                                                            roi_w=roi_w, 
                                                            roi_h=roi_h, 
                                                            distance_threshold=distance_threshold)

    # compute the euclidean distances
    P1, P2, P_diff = nusense.calculate_euclidean(img_ref_roi, img_roi) # not working
    nusense.visualize_euclidean_distances(P_diff, P1, P2)

    








