# Import required libraries
import cv2  # OpenCV for computer vision and image processing
import numpy as np  # NumPy for numerical operations
import pygame  # Pygame for creating the display window
from pygame.locals import *  # Pygame constants
from OpenGL.GL import *  # OpenGL core functionality
from OpenGL.GLU import *  # OpenGL utility functions
import mediapipe as mp  # Google's MediaPipe for hand tracking

# Initialize MediaPipe hand tracking
mp_hands = mp.solutions.hands
# Create a hand detector with 70% confidence threshold and tracking one hand
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils  # Utility for drawing hand landmarks

# Initialize webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Initialize Pygame and OpenGL
pygame.init()
display = (2000, 1400)  # Set display resolution
# Create a window with OpenGL support and double buffering
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
# Set up perspective projection (45-degree FOV, aspect ratio, near and far planes)
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
# Move the camera back 5 units to see the cube
glTranslatef(0.0, 0.0, -5)

# Define the 8 vertices of a cube in 3D space (x, y, z coordinates)
vertices = [
    (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),  # Front face
    (1, -1, 1), (1, 1, 1), (-1, 1, 1), (-1, -1, 1)       # Back face
]

# Define the 12 edges of the cube (pairs of vertex indices)
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Front face edges
    (4, 5), (5, 6), (6, 7), (7, 4),  # Back face edges
    (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
]

# Initialize transformation parameters for the cube
rotation = [0, 0]  # [x_rotation, y_rotation] in degrees
scale = 1.0       # Current scale factor
translation = [0, 0, 0]  # [x, y, z] translation

def draw_cube():
    """Draw the cube using OpenGL line drawing."""
    glBegin(GL_LINES)  # Start drawing lines
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])  # Draw each vertex of the edge
    glEnd()  # End drawing

def process_hand(frame):
    """
    Process a frame to detect and analyze hand gestures.
    Returns hand position, pinch distance, and finger count.
    """
    # Convert frame from BGR to RGB (MediaPipe requires RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame to detect hands
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        # Get the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get wrist position (landmark 0) as the hand's centroid
        wrist = hand_landmarks.landmark[0]
        cx = int(wrist.x * frame.shape[1])  # Convert to pixel coordinates
        cy = int(wrist.y * frame.shape[0])
        
        # Calculate pinch distance between thumb tip (4) and index tip (8)
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        pinch_distance = np.sqrt(
            ((index_tip.x - thumb_tip.x) * frame.shape[1])**2 +
            ((index_tip.y - thumb_tip.y) * frame.shape[0])**2
        )
        
        # Count raised fingers
        finger_count = 0
        tips = [4, 8, 12, 16, 20]  # Landmark indices for finger tips
        for tip in tips[1:]:  # Check all fingers except thumb
            # If tip is above the joint, finger is raised
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y:
                finger_count += 1
        # Check if thumb is extended
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[2].x:
            finger_count += 1
        
        # Draw hand landmarks and connections on the frame
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        return cx, cy, pinch_distance, finger_count
    
    return None, None, None, None

# Main application loop
running = True
while running:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Mirror the frame horizontally
    
    # Process hand gestures
    cx, cy, pinch_distance, finger_count = process_hand(frame)
    
    # Update cube transformations based on hand gestures
    if cx is not None and cy is not None:
        # Map hand position to cube rotation
        rotation[0] = (cy / frame.shape[0] - 0.5) * 360  # Y-axis rotation
        rotation[1] = (cx / frame.shape[1] - 0.5) * 360  # X-axis rotation
        
        # Scale cube based on pinch distance
        if pinch_distance is not None:
            scale = max(0.5, min(2.0, pinch_distance / 100))  # Limit scale between 0.5 and 2.0
        
        # Move cube when all fingers are raised
        if finger_count >= 4:
            translation[0] = (cx / frame.shape[1] - 0.5) * 2
            translation[1] = -(cy / frame.shape[0] - 0.5) * 2
    
    # Render the cube
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear the screen
    glLoadIdentity()  # Reset the transformation matrix
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)  # Reset perspective
    glTranslatef(translation[0], translation[1], -5)  # Apply translation
    glScalef(scale, scale, scale)  # Apply scaling
    glRotatef(rotation[0], 1, 0, 0)  # Apply X rotation
    glRotatef(rotation[1], 0, 1, 0)  # Apply Y rotation
    draw_cube()  # Draw the cube
    pygame.display.flip()  # Update the display
    
    # Show webcam feed with hand tracking
    cv2.imshow("Hand Tracking", frame)
    
    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:  # Quit on 'q' press
                running = False
            if event.key == pygame.K_r:  # Reset cube on 'r' press
                rotation = [0, 0]
                scale = 1.0
                translation = [0, 0, 0]
    
    # Handle OpenCV key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit on 'q' press
        running = False
    elif key == ord('r'):  # Reset cube on 'r' press
        rotation = [0, 0]
        scale = 1.0
        translation = [0, 0, 0]

# Clean up resources
cap.release()  # Release webcam
cv2.destroyAllWindows()  # Close OpenCV windows
pygame.quit()  # Quit Pygame