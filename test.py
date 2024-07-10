import cv2
import numpy as np
import os

# Function to detect faces using Haar cascades
def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces

# Function to analyze face shape (simplified)
def analyze_face_shape(face):
    # Placeholder for face shape analysis logic
    # Typically, this would involve geometric analysis or ML classifiers
    return "oval"  # Example, replace with actual logic

# Function to recommend hairstyles based on face shape
def recommend_hairstyle(face_shape):
    # Placeholder for hairstyle recommendation logic
    # Example: Dictionary mapping face shapes to hairstyles
    hairstyles = {
        "oval": "hairstyle1.png",
        "round": "hairstyle1.png",
        "square": "hairstyle1.png"
    }
    return hairstyles.get(face_shape, "hairstyle1.png")  # Default if face shape not recognized

# Function to overlay hairstyle on the detected face
def overlay_hairstyle(img, face, hairstyle_path):
    # Example: Resize and overlay hairstyle on the face
    x, y, w, h = face
    hairstyle = cv2.imread(hairstyle_path, -1)
    hairstyle = cv2.resize(hairstyle, (w, h))
    
    # Calculate position to center the hairstyle on the face
    x_offset = x - int(w * 0.1)  # Adjust x offset for hairstyle position
    y_offset = y - int(h * 0.5)  # Adjust y offset for hairstyle position

    # Ensure offsets stay within image bounds
    x_start = max(x_offset, 0)
    x_end = min(x_offset + hairstyle.shape[1], img.shape[1])
    y_start = max(y_offset, 0)
    y_end = min(y_offset + hairstyle.shape[0], img.shape[0])

    # Overlay the hairstyle on the face region
    for i in range(y_start, y_end):
        for j in range(x_start, x_end):
            alpha = hairstyle[i - y_start, j - x_start, 3] / 255.0
            img[i, j, :] = alpha * hairstyle[i - y_start, j - x_start, :3] + (1 - alpha) * img[i, j, :]

    return img

# Main function to process image and apply hairstyle overlay
def process_frame(frame):
    try:
        faces = detect_faces(frame)
        for face in faces:
            face_shape = analyze_face_shape(face)
            recommended_hairstyle = recommend_hairstyle(face_shape)
            frame = overlay_hairstyle(frame, face, recommended_hairstyle)
        
        cv2.imshow('Hairstyle Recommendation', frame)

    except Exception as e:
        print(f"Error processing frame: {e}")

# Initialize camera capture
cap = cv2.VideoCapture(0)  # 0 means the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from camera")
        break

    process_frame(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
