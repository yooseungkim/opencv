from face_recog_webcam import FaceRecog
import cv2
import numpy as np

webcam = FaceRecog()
video = cv2.VideoCapture("./data/background_vid.mp4")

w = round(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))

while True:
    frame, loc = webcam.get_frame()
    ret, img = video.read()

    if frame is None or img is None:
        break

    # show the frame

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = (30, 100, 100)
    upper_green = (100, 255, 255)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    area = np.count_nonzero(mask)
    ratio = frame.shape[0] / frame.shape[1]

    new_height = int(np.sqrt(area / ratio))
    new_width = int(new_height / ratio)

    top_most = None
    bottom_most = None
    left_most = None
    right_most = None
    hsv = cv2.resize(hsv, (0, 0),  fx=0.25, fy=0.25)
    for i in range(hsv.shape[0]):
        for j in range(hsv.shape[1]):
            h, s, v = hsv[i][j]
            if 30 <= h <= 100 and 100 <= s <= 255 and 100 <= v <= 255:
                if top_most == None:
                    top_most = i
                elif top_most > i:
                    top_most = i

                if bottom_most == None:
                    bottom_most = i
                elif bottom_most < i:
                    bottom_most = i

                if left_most == None:
                    left_most = j
                elif left_most > j:
                    left_most = j

                if right_most == None:
                    right_most = j
                elif right_most < j:
                    right_most = j

    horizontal = (left_most, right_most)
    vertical = (top_most, bottom_most)

    face = np.zeros_like(mask)
    if loc:
        face = frame[np.ix_(range(loc[0], loc[2]),
                            range(loc[3], loc[1]))]

    new_face = img.copy()
    if top_most and bottom_most and left_most and right_most:
        top_most = top_most * 4 + 30
        bottom_most *= 4
        left_most *= 4
        right_most *= 4

        # face = cv2.resize(
        #     face, (bottom_most - top_most, right_most - left_most))
        face = cv2.resize(
            face, (right_most - left_most, bottom_most - top_most))

        for i in range(top_most, bottom_most):
            for j in range(left_most, right_most):
                new_face[i][j] = face[i - top_most][j - left_most]

    cv2.imshow("Frame", new_face)
    out.write(frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

out.release()
video.release()
cv2.destroyAllWindows()
