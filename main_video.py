import cv2
# Assuming you have a custom module for face recognition
from custom_facerec import CustomFaceRecognizer

# Create a custom face recognition instance
face_recognizer = CustomFaceRecognizer()

# Load known face encodings from a directory
face_recognizer.load_face_encodings("known_faces/")

# Check available camera indices
# for i in range(10):
#     cap = cv2.VideoCapture(i)
#     if cap.isOpened():
#         print(f"Camera index {i} is available.")
#         cap.release()
#     else:
#         print(f"Camera index {i} is not available.")

# Use the default camera (index 0)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Detect and recognize faces
    face_locations, face_names = face_recognizer.recognize_faces(frame)

    # Draw bounding boxes and labels for recognized faces
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' key to exit
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
