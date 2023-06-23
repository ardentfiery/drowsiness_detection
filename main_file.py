import cv2
from tensorflow import keras
from keras.models import load_model
import numpy as np
from pygame import mixer
import mediapipe as mp

# mediapipe facedetection
mp_face_detection = mp.solutions.face_detection
score = 0
count = 0

# previously saved models
yawn_detector = load_model('models/yawn_detection_model.h5')
eye_detector = load_model("models/naya_aakha.h5")

# for alarm
mixer.init()
sound = mixer.Sound('sound/alarm.wav')
sound2 = mixer.Sound('sound/alarm.wav')

# capturing video
cap = cv2.VideoCapture(0)

# methods that crops mouth and eyes from frame and fed it to the pretrained models

def yawn_detection(frame, x, y):
    mouth = frame[y-45:y+50, x-65:x+60]
    cv2.rectangle(frame, (x - 65, y - 45), 
                  (x + 60, y + 50), (255, 0, 255), 2)
    gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (80, 80))
    gray = gray / 255
    gray = np.expand_dims(gray, axis=0)
    predicted = yawn_detector.predict(gray)[0][0]
    return predicted

def eyes_detection(frame, x, y, a, b):
    eyes = frame[y-22:b+17, x-37:a+37]
    cv2.rectangle(frame, (eye1_x - 45, eye1_y - 30),  # drawing rectange with absolute coordinate
                  (eye2_x + 45, eye2_y + 25), (0, 255, 255), 2)
    gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (80, 80))
    gray = gray / 255
    gray = np.expand_dims(gray, axis=0)
    predicted = eye_detector.predict(gray)[0][0]
    return predicted


# mediapipe face detection starts
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, frame = cap.read()

        # extracting height width so that paxi rectange ko absolute position calculate garna sakam
        height, width, channels = frame.shape
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame)

        # frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:

                # face detection ko rectange ko relative coordinate taaneko
                relative_data = detection.location_data.relative_bounding_box
               # mp_drawing.draw_detection(frame, detection)

                # calculating absolute coordinate
                axmin = int(relative_data.xmin * width)
                aymin = int(relative_data.ymin * height)
                awidth = int(relative_data.width * width)
                aheight = int(relative_data.height * height)
                top_left_x = axmin
                top_left_y = aymin
                bottom_right_x = axmin + awidth
                bottom_right_y = aymin + aheight

                cv2.rectangle(frame, (top_left_x, top_left_y),  # drawing rectange with absolute coordinate
                              (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

                aye = detection.location_data.relative_keypoints[0:2]
                # calculating coordinates for eyes
                eye1_x = int(aye[0].x * width)
                eye1_y = int(aye[0].y * height)
                eye2_x = int(aye[1].x * width)
                eye2_y = int(aye[1].y * height)

                # calculating coordinates for mouth
                mukh = detection.location_data.relative_keypoints[3]
                mukh_x = int(mukh.x * width)
                mukh_y = int(mukh.y * height)

                # detection starts here
                yawn_pred = yawn_detection(frame, mukh_x, mukh_y)

                if yawn_pred > 0.75:
                    text = "Yawn"
                    count += 1
                elif yawn_pred > 0:
                    text = "No Yawn"
                cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_DUPLEX,
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)

                eyes_pred = eyes_detection(
                    frame, eye1_x, eye1_y, eye2_x, eye2_y)
                if eyes_pred < 0.6:
                    text = "Close"
                    score = score+1
                    if (score > 10):
                        try:
                            sound.play()
                            cv2.putText(frame, "Drowsy!!", (240, 65), cv2.FONT_HERSHEY_DUPLEX,
                                        1, (0, 0, 255), 1, cv2.LINE_AA)

                        except:
                            pass
                else:
                    text = "Open"
                    score = score - 2
                    if (score < 0):
                        score = 0
                cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_DUPLEX,
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # taking 15 counts as 1 yawn so when user yawns for 4 times he will be alerted
        yawn_count = int(count/15)
        if yawn_count % 5 == 0:
            sound2.play()
            cv2.putText(frame, "Yawn Drowsy!!", (10, 55), cv2.FONT_HERSHEY_DUPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)
            sound2.pause()
                
        # Flip the frame horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Detection', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
