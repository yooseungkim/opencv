# face_recog.py

import face_recognition
import cv2
import camera
import os
import numpy as np


class FaceRecog():
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.

        # To use webcam
        self.camera = camera.VideoCamera()

        # Save Video
        # self.videowriter = cv2.VideoWriter(
        #     "recog.mp4", cv2.VideoWriter_fourcc(
        #         *'MP4V'), 30, (int(self.camera.get(3)), int(self.camera.get(4)))
        # )

        self.known_face_encodings = []
        self.known_face_names = []

        # Load sample pictures and learn how to recognize it.
        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

    def __del__(self):
        del self.camera

    def get_frame(self):
        frame = self.camera.get_frame()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if self.process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            self.face_locations = face_recognition.face_locations(
                rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(
                rgb_small_frame, self.face_locations)

            self.face_names = ["UNKNOWN"] * len(self.face_encodings)

        self.process_this_frame = not self.process_this_frame

        # Display the results
        if self.face_locations:
            top, right, bottom, left = self.face_locations[0]
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # # Draw a box around the face
            # cv2.rectangle(frame, (left, top),
            #               (right, bottom), (0, 0, 255), 2)

            # # Draw a label with a name below the face
            # cv2.rectangle(frame, (left, bottom - 35),
            #               (right, bottom), (0, 0, 255), cv2.FILLED)
            # font = cv2.FONT_HERSHEY_DUPLEX
            # cv2.putText(frame, "unknown", (left + 6, bottom - 6),
            #             font, 1.0, (255, 255, 255), 1)

        return frame, ((int(top * 1.05), int(right * 0.95), int(bottom * 0.95), int(left * 1.05)) if self.face_locations else None)

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

    while True:
        frame, loc = face_recog.get_frame()

        if frame is None:
            break

        # show the frame
        cv2.imshow("Frame", frame)
        # face_recog.videowriter.write(frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    # face_recog.videowriter.release()
    cv2.destroyAllWindows()
    print('finish')
