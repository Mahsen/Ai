import cv2
import numpy as np

# Load the image
image = cv2.imread('D:/Project/NewStructure/Ai/BoardDiff/pcb_image.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Thresholding to get binary image
_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through contours
components = {'resistor': 0, 'capacitor': 0, 'inductor': 0, 'transistor': 0, 'DIP_IC': 0}  # You can add more components as needed
for contour in contours:
    # Approximate the contour
    approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)

    # Calculate the number of vertices
    vertices = len(approx)

    # Define conditions for different components based on number of vertices
    if vertices > 5:  # Resistor
        components['resistor'] += 1
        # Draw a bounding box around the component
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    elif vertices == 4:  # Capacitor, Inductor, Transistor, DIP IC
        # Calculate the aspect ratio
        aspect_ratio = float(w) / h
        if aspect_ratio < 1:  # DIP IC
            components['DIP_IC'] += 1
            # Draw a bounding box around the component
            cv2.drawContours(image, [approx], 0, (255, 0, 0), 2)
        else:  # Capacitor, Inductor, Transistor
            components['capacitor'] += 1
            # Draw a bounding box around the component
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    elif vertices == 3:  # Transistor
        components['transistor'] += 1
        # Draw a bounding box around the component
        cv2.drawContours(image, [approx], 0, (255, 0, 0), 2)
    # You can add more conditions based on the shapes of components

# Print the number of each component
print("Number of resistors:", components['resistor'])
print("Number of capacitors:", components['capacitor'])
print("Number of inductors:", components['inductor'])
print("Number of transistors:", components['transistor'])
print("Number of DIP ICs:", components['DIP_IC'])

# Display the image with bounding boxes
cv2.imshow('Detected Components', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
