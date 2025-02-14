# face_recog.py
import face_recognition
import cv2
import camera
import os
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()

parser.add_argument("--source", dest="path", required=True)
parser.add_argument("--verbose", dest="verbose", default=False)
parser.add_argument("--target", dest="target", default=None)
args = parser.parse_args()
path = args.path


class FaceRecog():
    def __init__(self):

        self.image = cv2.imread(path)
        self.name = path.split("/")[-1][:-4]
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

    def get_frame(self):

        frame = self.image

        rgb_frame = frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if self.process_this_frame:
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

                self.face_names.append(name)

        self.process_this_frame = not self.process_this_frame

        found = None
        # Display the results
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):

            name = "".join(i for i in name if not i.isdigit())

            center = ((right + left) // 2, (top + bottom) // 2)
            if args.verbose:
                print(f"{name}: ({center[0]}, {center[1]})")

            cv2.line(frame, center, center, (255, 0, 0), 10)

            # Draw a box around the face
            cv2.rectangle(frame, (left, top),
                          (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35),
                          (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        font, 1.0, (255, 255, 255), 1)

        cv2.imwrite(f"recog_video/{self.name}.jpg", frame)

        return frame

    def get_jpg_bytes(self):
        frame = self.get_frame()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()


if __name__ == '__main__':
    face_recog = FaceRecog()
    print(face_recog.known_face_names)
    frame = face_recog.get_frame()

    cv2.destroyAllWindows()
    print('finish')
