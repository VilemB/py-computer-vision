import cv2
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Pygame and OpenGL setup
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)

# Cube vertices and edges
vertices = [
    (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
    (1, -1, 1), (1, 1, 1), (-1, 1, 1), (-1, -1, 1)
]
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

# Cube transformation parameters
rotation = [0, 0]
scale = 1.0
translation = [0, 0, 0]

def draw_cube():
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

def process_hand(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get centroid (wrist landmark)
        wrist = hand_landmarks.landmark[0]
        cx = int(wrist.x * frame.shape[1])
        cy = int(wrist.y * frame.shape[0])
        
        # Get pinch distance (thumb tip to index tip)
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        pinch_distance = np.sqrt(
            ((index_tip.x - thumb_tip.x) * frame.shape[1])**2 +
            ((index_tip.y - thumb_tip.y) * frame.shape[0])**2
        )
        
        # Estimate finger count (fingers raised)
        finger_count = 0
        tips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky
        for tip in tips[1:]:  # Exclude thumb
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y:
                finger_count += 1
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[2].x:  # Thumb extended
            finger_count += 1
        
        # Draw landmarks
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        return cx, cy, pinch_distance, finger_count
    
    return None, None, None, None

running = True
while running:
    # Process webcam frame
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    
    # Process hand
    cx, cy, pinch_distance, finger_count = process_hand(frame)
    
    # Update cube transformations
    if cx is not None and cy is not None:
        # Map hand position to rotation
        rotation[0] = (cy / frame.shape[0] - 0.5) * 360  # Y-axis rotation
        rotation[1] = (cx / frame.shape[1] - 0.5) * 360  # X-axis rotation
        
        # Scale based on pinch distance
        if pinch_distance is not None:
            scale = max(0.5, min(2.0, pinch_distance / 100))
        
        # Translate if open hand (5 fingers)
        if finger_count >= 4:
            translation[0] = (cx / frame.shape[1] - 0.5) * 2
            translation[1] = -(cy / frame.shape[0] - 0.5) * 2
    
    # Render cube
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(translation[0], translation[1], -5)
    glScalef(scale, scale, scale)
    glRotatef(rotation[0], 1, 0, 0)
    glRotatef(rotation[1], 0, 1, 0)
    draw_cube()
    pygame.display.flip()
    
    # Display webcam feed
    cv2.imshow("Hand Tracking", frame)
    
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            if event.key == pygame.K_r:  # Reset cube
                rotation = [0, 0]
                scale = 1.0
                translation = [0, 0, 0]
    
    # Handle OpenCV key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        running = False
    elif key == ord('r'):
        rotation = [0, 0]
        scale = 1.0
        translation = [0, 0, 0]

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()