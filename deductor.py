
import cv2
import numpy as np
from plyer import notification
import threading
from joblib import load
import mediapipe as mp
from skimage.feature import hog
import time
from twilio.rest import Client

import os

import smtplib
from email.message import EmailMessage

SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_ADDRESS = 'rakeshkumarsenthilkrct@gmail.com'
EMAIL_PASSWORD = 'pkls yhus ehwh zbxv'
recipient_email = 'rakeshkumarsenthil@gmail.com'



# save_dir = 'D:\detect_posture_2\detect_posture\captured_images'
# os.makedirs(save_dir, exist_ok=True)
# i=0

# Load the MediaPipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load posture detection model
posture_model = load('trained_bad.pkl')

# Function to preprocess image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    return resized

# Function to send notification for bad posture
# def notify():
#                 account_sid = "AC5d3762758bd3d1f5af922790bab87c8a"
#                 auth_token = "17259529b3e3d09badb40d52603aaa00"
#                 client = Client(account_sid, auth_token)

#                 call = client.calls.create(
#                 url="http://demo.twilio.com/docs/voice.xml",
#                 to="+917695980388",
#                 from_="+16235525213"
#                 )
#                 print(call.sid)
                # message = client.messages.create(
                # body='Your Baby in Danger!',
                # to="+917695980388",
                # from_="+16235525213"
                # # media_url=[media_url]
                # )

                # print(f'Message SID: {message.sid}')


# Function to reset the notification flag after a timeout
def reset_notification_flag():
    global send_notification
    send_notification = False

# Initialize notification flag and posture tracking variables
send_notification = False
bad_posture_start_time = None
posture_value = "good"

# Function to start posture detection
def send_email(image_path):
    msg = EmailMessage()
    msg['Subject'] = 'Here is an image for you!'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = recipient_email
    msg.set_content('Please find the attached image.')

    with open(image_path, 'rb') as img:
        img_data = img.read()
        img_name = os.path.basename(image_path)
        msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=img_name)

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)

    print('Email sent successfully!')

def startPostureDetection():
    global send_notification, bad_posture_start_time, posture_value
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Cannot receive frame (stream end?). Exiting ...")
                break

            # Preprocess frame
            preprocessed_frame = preprocess_image(frame)

            # Extract HOG features
            fd = hog(preprocessed_frame, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
            
            # Predict posture
            prediction = posture_model.predict(fd.reshape(1, -1))
            posture_value = "bad" if prediction == 0 else "good"

            # Set font and text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (0, 0, 255) if posture_value == "bad" else (0, 255, 0)

            # Check for bad posture and track the duration
            if posture_value == 'bad':
                if bad_posture_start_time is None:
                    bad_posture_start_time = time.time()
                elif time.time() - bad_posture_start_time > 2 and not send_notification:
                    ############# if pose is wrong
                    # print("hi")
                    # notify()
                    ### call
                    account_sid = "AC5d3762758bd3d1f5af922790bab87c8a"
                    auth_token = "f97a267556ee595d999e07143229665e"
                    client = Client(account_sid, auth_token)

                    call = client.calls.create(
                    url="http://demo.twilio.com/docs/voice.xml",
                    to="+917695980388",
                    from_="+16235525213"
                    )
                    print(call.sid)
                    
                    message = client.messages.create(
                    body='Your Baby in Danger!',
                    to="+917695980388",
                    from_="+16235525213"
                    )
                    print(f'Message SID: {message.sid}')
                    ##### mail
                    image_path = 'captured_frame.jpg'
                    cv2.imwrite(image_path, frame)
                    threading.Thread(target=send_email, args=(image_path,)).start()


                    send_notification = True
                    threading.Timer(10, reset_notification_flag).start()  # Start timer to reset the flag after 10 seconds
            else:
                bad_posture_start_time = None  # Reset the timer if posture is good
            
            # Overlay the predicted posture text on the frame
            cv2.putText(frame, f"Posture: {posture_value}", (10, 30), font, font_scale, font_color, 2)
            
            # Convert the frame to RGB for landmark detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect landmarks
            results = pose.process(rgb_frame)
            
            # Draw Pose landmarks on the frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image=frame, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)

            # Display the frame with the predicted posture and landmarks
            cv2.imshow('Posture Detection', frame)

            # Check for user input to quit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

# Start posture detection
startPostureDetection()
