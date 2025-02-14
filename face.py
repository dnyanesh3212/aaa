import face_recognition
import cv2
import os
import numpy as np

# Create a directory to store known faces
if not os.path.exists("students"):
    os.makedirs("students")

# Function to capture a student's face and save the encoding
def capture_face(student_name):
    video_capture = cv2.VideoCapture(0)  # Use the first camera
    
    # Allow the camera to warm up
    print("Capturing image for", student_name)
    
    while True:
        ret, frame = video_capture.read()
        
        # Detect faces in the frame
        face_locations = face_recognition.face_locations(frame)
        
        # If faces are detected
        if len(face_locations) > 0:
            face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
            np.save(f"students/{student_name}_encoding.npy", face_encoding)
            print(f"Face encoding for {student_name} saved.")
            break
        
        # Display the captured frame
        cv2.imshow("Capture Face", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

# Example usage: capture faces for two students
capture_face("Alice")
capture_face("Bob")
