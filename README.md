# Python Computer Vision Hand Tracking Test

A simple project for testing out computer vision concepts with Python, specifically hand tracking to control a 3D cube with Pygame and PyOpenGL.

## Features

*   Real-time hand tracking using webcam input via MediaPipe.
*   Control a 3D cube's rotation, scale, and translation using hand gestures:
    *   Hand position: Controls rotation.
    *   Pinch gesture (thumb and index finger): Controls scaling.
    *   Open hand (4 or 5 fingers detected): Controls translation.
*   Uses OpenCV for video capture and image processing.
*   Uses Pygame for the window and event handling, and PyOpenGL for 3D rendering.

## Prerequisites

*   Python 3 (e.g., 3.9+)
*   A webcam

## Setup & Running

1.  **Clone the repository (if you've put it on GitHub) or ensure all files (`app.py`, `requirements.txt`) are in the same directory.**

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate   # On Windows
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python app.py
    ```

## Controls

*   **Move your hand** in front of the webcam: Rotates the cube.
*   **Pinch thumb and index finger**: Scales the cube.
*   **Show an open hand (4-5 fingers detected)** and move it: Translates the cube.
*   Press **'q'** (in the webcam window or the Pygame window): Quits the application.
*   Press **'r'** (in the webcam window or the Pygame window): Resets the cube's transformation.

## Built With

*   [OpenCV](https://opencv.org/)
*   [NumPy](https://numpy.org/)
*   [Pygame](https://www.pygame.org/)
*   [PyOpenGL](http://pyopengl.sourceforge.net/)
*   [MediaPipe](https://mediapipe.dev/)