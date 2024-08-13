import cv2
import numpy as np

def apply_sobel_with_bilateral_filter_to_image(input_image_path, output_image_path, roi_x, roi_y, roi_w, roi_h, distance_threshold=50):
    # Load the image
    image = cv2.imread(input_image_path)
    
    if image is None:
        print("Error: Could not load image.")
        return
    
    # Apply ROI
    roi_image = image[roi_y:roi_y + roi_h, roi_x: roi_x + roi_w]
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    
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
                cv2.drawContours(roi_image, [approx], 0, (0, 255, 0), 2)
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
                cv2.line(roi_image, tuple(point1[0]), tuple(closest_point[0]), (255, 255, 255), 2)
                
                # Calculate the midpoint
                midpoint = ((point1[0][0] + closest_point[0][0]) // 2, (point1[0][1] + closest_point[0][1]) // 2)
                
                # Draw a small circle at the midpoint
                cv2.circle(roi_image, midpoint, 3, (255, 0, 0), -1)
    
    # Save the processed image
    cv2.imwrite(output_image_path, roi_image)
    
    # Display the image with highlighted quadrilaterals and connected lines
    cv2.imshow('Quadrilateral Tracking', roi_image)
    cv2.imshow('Binary Image', closed_binary)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

input_image_path = r'D:/Experiments/2/Photos_Defisheyed/x1_z2.jpg'
output_image_path = r'D:/Experiments/2/Photos_Output/x1_z2.jpg'
#roi_x, roi_y, roi_w, roi_h = 480, 260, 1145, 800
roi_x, roi_y, roi_w, roi_h = 390, 203, 1335, 1000
distance_threshold = 50 
apply_sobel_with_bilateral_filter_to_image(input_image_path, output_image_path, roi_x, roi_y, roi_w, roi_h, distance_threshold=distance_threshold)
