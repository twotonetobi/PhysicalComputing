"""
MediaPipe Tracking Example
adopted Tutorial Example from MediaPipe Documentation
https://google.github.io/mediapipe/solutions/pose#python-solution-api

Physical Computing
WS 22/23 HAW Hamburg
Presentation Asset
Tobias Wursthorn
1629762

"""


import queue

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Get image shape
    shape = image.shape
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Draw the pose annotation on the image.
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Using Nose ID
    Id_1 = 0
    # Get XYZ Values
    x1 = results.pose_landmarks.landmark[Id_1].x
    x1_ = str(x1)[0:5]
    y1 = results.pose_landmarks.landmark[Id_1].y
    y1_ = str(y1)[0:5]
    z1 = results.pose_landmarks.landmark[Id_1].z
    z1_ = str(z1)[0:5]
    text = 'x' + x1_ + ' y' + y1_ + ' z' + z1_
    # Map the values into image shape
    relative_x1 = int(x1 * shape[1])
    relative_y1 = int(y1 * shape[0])
    cv2.circle(image, (relative_x1, relative_y1), radius=3, color=(255, 0, 0), thickness=2)
    cv2.putText(image, text=text, org=(relative_x1, relative_y1), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1,
                color=(255, 0, 255), thickness=1)
    # Do not flip the image
    # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    cv2.imshow('MediaPipe Pose',image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
