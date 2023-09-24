import face_recognition
import cv2
import os
import glob
import numpy as np


class CustomFaceRecognizer:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Adjust the frame size for improved performance
        self.frame_scaling = 0.25

    def load_face_encodings(self, directory_path):
        """
        Load face encodings from a directory.
        :param directory_path: Path to the directory containing face images.
        """
        # Get a list of image files in the directory
        image_files = glob.glob(os.path.join(directory_path, "*.*"))

        print("{} face encodings found.".format(len(image_files)))

        # Process each image and store its encoding and name
        for image_path in image_files:
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Extract the filename from the full path
            base_name = os.path.basename(image_path)
            file_name, _ = os.path.splitext(base_name)

            # Compute the face encoding
            face_encoding = face_recognition.face_encodings(rgb_image)[0]

            # Store the file name and face encoding
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(file_name)

        print("Face encodings loaded")

    def recognize_faces(self, frame):
        # Resize the frame for faster processing
        small_frame = cv2.resize(
            frame, (0, 0), fx=self.frame_scaling, fy=self.frame_scaling)

        # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        recognized_names = []
        for face_encoding in face_encodings:
            # Check if the face matches any of the known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Find the best match by comparing face distances
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            recognized_names.append(name)

        # Convert face locations to match the resized frame
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_scaling

        return face_locations.astype(int), recognized_names
