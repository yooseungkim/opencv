# face_recog.py
import face_recognition
import cv2
import camera
import os
import numpy as np
import argparse
import time

# parser = argparse.ArgumentParser()

# parser.add_argument("--source", dest="path", required=True)
# parser.add_argument("--verbose", dest="verbose", default=False)
# parser.add_argument("--target", dest="target", default=None)
# args = parser.parse_args()
# path = args.path


class FaceRecog():
    def __init__(self):

        # self.image = cv2.imread(path)
        # self.name = path.split("/")[-1][:-4]
        self.name = "test"
        self.known_face_encodings = []
        self.known_face_names = []

        # Load sample pictures and learn how to recognize it.
        dirname = 'knowns'
        files = os.listdir(dirname)
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext == '.jpg':
                self.known_face_names.append(name)
                pathname = os.path.join(dirname, filename)
                img = face_recognition.load_image_file(pathname)
                face_encoding = face_recognition.face_encodings(img)[0]
                self.known_face_encodings.append(face_encoding)

        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

    def get_frame(self, image):

        # frame = self.image

        # rgb_frame = frame[:, :, ::-1]
        rgb_frame = image[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        self.face_locations = face_recognition.face_locations(
            rgb_frame)
        self.face_encodings = face_recognition.face_encodings(
            rgb_frame, self.face_locations)

        self.face_names = []
        for face_encoding in self.face_encodings:
            # See if the face is a match for the known face(s)
            distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding)
            min_value = min(distances)

            name = "Unknown"
            if min_value < 0.6:
                index = np.argmin(distances)
                name = self.known_face_names[index]

                return True

        return False


if __name__ == '__main__':
    image1 = cv2.imread(
        "/Users/yooseungkim/Downloads/nego/Screenshot 2022-12-15 at 4.31.11 PM.png")
    face_recog = FaceRecog()
    print(face_recog.known_face_names)
    index = face_recog.get_frame(image1)
    print(index)

    # cv2.destroyAllWindows()
    print('finish')
