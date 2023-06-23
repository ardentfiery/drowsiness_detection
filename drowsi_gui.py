import tkinter as tk
import cv2
import mediapipe as mp
from tensorflow import keras
from keras.models import load_model
import numpy as np
from pygame import mixer
from PIL import Image, ImageTk

class DrowsinessDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drowsiness Detection")
        self.root.geometry('1000x720')
        self.root.configure(background='#27374D')

        welcome_text = tk.Label(self.root, text="Welcome To Drowsiness Detection System", fg="white", bg="#27374D")
        welcome_text.pack(pady=(20,10))
        welcome_text.config(font=('Segoe UI', 24))
        welcome_text = tk.Label(self.root, text='"' + "Road safety is state of mind, accident is absence of mind."+'"', fg="white", bg="#27374D",font=('Segoe UI', 12, "italic"))
        welcome_text.pack()

        button_style = {
            'font': ('Arial', 14),
            'bg': '#AFD3E2',
            'fg': 'black',
            'activebackground': '#9DB2BF',
            'activeforeground': 'white',
            'bd': 0,
            'highlightthickness': 0,
            # 'relief': 'flat',
            'width': 20,
        }

        # Start and Stop Button
        self.start_button = tk.Button(root, text="Start",**button_style, command=self.start_camera)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(root, text="Stop", state=tk.DISABLED,**button_style, command=self.stop_camera,)
        self.stop_button.pack(pady=10)

        # Label for video
        self.video_label = tk.Label(root,bg="#27374D", highlightthickness=2, highlightbackground="white")
        self.video_label.place(x=210,y=230,width=640, height=480)

        # Initializing openvc and mediapipe and score and count for eyes and yawn and first_time for second time start button clicking problem solution
        self.cap = cv2.VideoCapture(0)
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.score = 0
        self.count = 0
        self.first_time = 0

        # Loading pretrained saved model
        self.yawn_detector = load_model('models/yawn_detection_model.h5')
        self.eye_detector = load_model("models/naya_aakha.h5")

        # For alarm
        mixer.init()
        self.sound = mixer.Sound('sound/alarm.wav')

        # For checking camera state { open or close }
        self.camera_running = False

        # Starting the tkinter main loop
        self.root.mainloop()

    #method for capturing frame, perform detection and fireup alarm and display
    def process_camera(self):
        if self.camera_running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.mp_face_detection.process(frame)

                if results.detections:
                    for detection in results.detections:
                        relative_data = detection.location_data.relative_bounding_box
                        height, width, channels = frame.shape
                        # calculating actual coordinate for face rectangle
                        axmin = int(relative_data.xmin * width)
                        aymin = int(relative_data.ymin * height)
                        awidth = int(relative_data.width * width)
                        aheight = int(relative_data.height * height)

                        # calculating actual coordinate for eyes rectangle
                        aye = detection.location_data.relative_keypoints[0:2]
                        eye1_x = int(aye[0].x * width)
                        eye1_y = int(aye[0].y * height)
                        eye2_x = int(aye[1].x * width)
                        eye2_y = int(aye[1].y * height)

                        # calculating actual coordinate for mouth rectangle
                        mukh = detection.location_data.relative_keypoints[3]
                        mukh_x = int(mukh.x * width)
                        mukh_y = int(mukh.y * height)

                        # cropping mouth from frame to feed into the pretrained model
                        mouth = frame[mukh_y-45:mukh_y+50, mukh_x-65:mukh_x+60]

                        if mouth.size == 0:
                            continue
                        cv2.rectangle(frame, (mukh_x - 65, mukh_y - 45),
                                      (mukh_x + 60, mukh_y + 50), (255, 0, 255), 2)

                        gray = cv2.cvtColor(mouth, cv2.COLOR_RGB2GRAY)
                        gray = cv2.resize(gray, (80, 80))
                        gray = gray / 255
                        gray = np.expand_dims(gray, axis=0)
                        yawn_pred = self.yawn_detector.predict(gray)[0][0]

                        if yawn_pred == 100:
                            text = "Mouth Not Detected"
                        elif yawn_pred > 0.75:
                            text = "Yawn"
                            self.count += 1
                        elif yawn_pred > 0:
                            text = "No Yawn"
                        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_DUPLEX,
                                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        yawn_count = int(self.count/10)
                        if yawn_count % 5 == 0:
                            self.sound.play()
                            cv2.putText(frame, "Yawn Drowsy!!", (10, 55), cv2.FONT_HERSHEY_DUPLEX,
                            0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            self.sound.pause()
                        
                        # cropping eyes from frame to feed int pretrained model
                        eyes = frame[eye1_y-22:eye2_y+17, eye1_x-37:eye2_x+37]
                        if eyes.size == 0:
                            continue
                        cv2.rectangle(frame, (eye1_x - 45, eye1_y - 30),
                                      (eye2_x + 45, eye2_y + 25), (0, 255, 255), 2)
                        gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
                        gray = cv2.resize(gray, (80, 80))
                        gray = gray / 255
                        gray = np.expand_dims(gray, axis=0)
                        eyes_pred = self.eye_detector.predict(gray)[0][0]

                        if eyes_pred == 100:
                            text = "Eyes Not Detected"
                        elif eyes_pred < 0.6:
                            text = "Close"
                            self.score = self.score+1
                            if self.score > 10:
                                try:
                                    self.sound.play()
                                    cv2.putText(frame, "Drowsy!!", (240, 65), cv2.FONT_HERSHEY_DUPLEX,
                                                    1, (0, 0, 255), 1, cv2.LINE_AA)
                                except:
                                    pass
                        else:
                            text = "Open"
                            self.sound.stop()
                            self.score = self.score - 2
                            if self.score < 0:
                                self.score = 0
                        cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_DUPLEX,
                                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

                        #display frame in gui
                        img = Image.fromarray(frame)
                        img = img.resize((640, 480))
                        imgtk = ImageTk.PhotoImage(image=img)
                        self.video_label.imgtk = imgtk
                        self.video_label.configure(image=imgtk)
                else:
                    img = Image.fromarray(frame)
                    img = img.resize((640, 480))
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)

            self.video_label.after(10, self.process_camera)
        else:
            self.cap.release()
            self.video_label.imgtk = None
            self.video_label.configure(image="")
            self.sound.stop()

    # this method firesup when start button is clicked
    def start_camera(self):
        if not self.camera_running:
            self.camera_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.process_camera()
            if self.first_time > 0:
                self.cap = cv2.VideoCapture(0)
            self.first_time += 1

    # this method firesup when stop button is clicked
    def stop_camera(self):
        if self.camera_running:
            self.camera_running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.score = 0;
            self.count = 0;
            self.cap.release()
            self.video_label.imgtk = None
            self.video_label.configure(image="")
            self.sound.stop()

root = tk.Tk()
app = DrowsinessDetectionApp(root)