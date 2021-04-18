import mediapipe as mp
import cv2
import numpy as np
# import pyautogui as p
import time
from directkeys import W, A, S, D, PressKey, ReleaseKey

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_elbowangle(hip, shoulder, elbow):
    hip = np.array(hip)  # hip
    shoulder = np.array(shoulder)  # elbow
    elbow = np.array(elbow)  # wrist

    # 1 is y, 0 is x. Gets the vector from bc to ba
    radians = np.arctan2(hip[1]-shoulder[1], hip[0]-shoulder[0]) - \
        np.arctan2(elbow[1]-shoulder[1], elbow[0]-shoulder[0])
    elbowangle = np.abs(radians*180.0/np.pi)

    if elbowangle > 180.0:
        elbowangle = 360-elbowangle

    return elbowangle


def calculate_wristangle(shoulder, elbow, wrist):
    shoulder = np.array(shoulder)  # hip
    elbow = np.array(elbow)  # elbow
    wrist = np.array(wrist)  # wrist

    # 1 is y, 0 is x. Gets the vector from bc to ba
    radians = np.arctan2(shoulder[1]-elbow[1], shoulder[0]-elbow[0]) - \
        np.arctan2(wrist[1]-elbow[1], wrist[0]-elbow[0])
    wristangle = np.abs(radians*180.0/np.pi)

    if wristangle > 180.0:
        wristangle = 360-wristangle

    return wristangle


def calculate_shoulder_angle(shoulder1, shoulder2):
    shoulder1 = np.array(shoulder1)  # hip
    shoulder2 = np.array(shoulder2)  # elbow

    # 1 is y, 0 is x. Gets the vector from bc to ba
    radians = np.arctan2(shoulder1[1]-shoulder2[1], shoulder1[0]-shoulder2[0])
    shoulder_angle = radians*180.0/np.pi
    if shoulder_angle > 180.0:
        shoulder_angle = 360-shoulder_angle

    return shoulder_angle


cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
#         frame = cv2.flip(frame,2)
        # Change BGR color to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # make detections
        results = pose.process(image)

        # change RGB to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            leftshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            lefthip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            leftelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            leftwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            rightshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            righthip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            rightelbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            rightwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Caclulate shoulder angle
            left_shoulder_angle = calculate_shoulder_angle(
                leftshoulder, rightshoulder)
            # right_shoulder_angle = calculate_shoulder_angle(
            #     rightshoulder, leftshoulder)

            # Caclulate Elbow Angle
            left_elbow_angle = calculate_elbowangle(
                lefthip, leftshoulder, leftelbow)
            right_elbow_angle = calculate_elbowangle(
                righthip, rightshoulder, rightelbow)

            # Calculate Wrist Angle
            left_wrist_angle = calculate_wristangle(
                leftshoulder, leftelbow, leftwrist)
            right_wrist_angle = calculate_wristangle(
                rightshoulder, rightelbow, rightwrist)

            # visualize
            # left elbow
            cv2.putText(image, str(left_shoulder_angle), tuple(np.multiply(leftshoulder, [640, 480]).astype(
                int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.putText(image, str(rightshoulder_angle), tuple(np.multiply(rightshoulder, [640, 480]).astype(
            #     int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            # right elbow
            cv2.putText(image, str(rightelbow[0]), tuple(np.multiply(rightelbow, [640, 480]).astype(
                int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            # left wrist
            cv2.putText(image, str(leftwrist[0]), tuple(np.multiply(leftwrist, [640, 480]).astype(
                int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            # right wrist
            cv2.putText(image, str(rightwrist[0]), tuple(np.multiply(rightwrist, [640, 480]).astype(
                int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            if left_elbow_angle > 90 and right_elbow_angle > 90 and left_wrist_angle > 90 and right_wrist_angle > 90:
                ReleaseKey(A)
                ReleaseKey(D)
                ReleaseKey(S)
                PressKey(W)
                # time.sleep(1)

            elif left_elbow_angle > 90 and right_elbow_angle > 90 and left_wrist_angle < 90 and right_wrist_angle < 90:
                ReleaseKey(W)
                PressKey(S)
                # p.keyDown('s')
                # p.keyUp('w')
                # time.sleep(0.25)
            else:
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(S)
                ReleaseKey(D)
            print(left_shoulder_angle)
            if left_shoulder_angle < -15:
                ReleaseKey(A)
                PressKey(D)
            elif left_shoulder_angle > 15:
                ReleaseKey(D)
                PressKey(A)
            else:
                ReleaseKey(A)
                ReleaseKey(D)

        except:
            ReleaseKey(A)
            ReleaseKey(W)
            ReleaseKey(S)
            ReleaseKey(D)
            pass

#         #draw face landmarks
#         mp_drawing.draw_landmarks(image, results.face_landmarks,mp_holistic.FACE_CONNECTIONS)

#         #right hand
#         mp_drawing.draw_landmarks(image, results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
#                                   mp_drawing.DrawingSpec(color=(0,255,0),thickness=2,circle_radius=2),
#                                   mp_drawing.DrawingSpec(color=(0,0,255),thickness=2,circle_radius=2))

#         #left hand
#         mp_drawing.draw_landmarks(image, results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)

        # pose
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # image show
        cv2.imshow('Holistic Model Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

ReleaseKey(A)
ReleaseKey(W)
ReleaseKey(S)
ReleaseKey(D)

cap.release()
cv2.destroyAllWindows()
