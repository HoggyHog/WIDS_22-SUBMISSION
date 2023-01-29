from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np 
import cv2
from playsound import playsound
import time

# Load the respective emotions and gesture Models

emotion_model = load_model("emotions.h5")


# Load the Haar Cascade classifier for face detection

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Sort the emotions and gesture labels

emotion_label = ['Angry', 'Disgust','Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


# Analyze users choice 
# 1 for emotions detector
# 2 for gestures detector

print("""
Enter 1 to begin
""")

choice = int(input())

if choice == 1:

    """
        In this choice we will be running the emotions model.
        While webcam is detected we will read the frames and
        then we will draw a rectangle box when the haar cascade
        classifier detects a face. We will convert the facial 
        image into a grayscale of dimensions 48, 48 similar to
        the trained images for better predictions.
        The Prediction is only made when the np.sum detects at
        least one face. The keras commands img_to_array converts
        the image to array dimensions and in case more images are
        detected we expand the dimensions.
        The Predictions are made according to the labels and the 
        recordings will be played accordingly.
    """

    cap = cv2.VideoCapture(0)

    while True:
        # Grab a single frame of video
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        # rect,face,image = face_detector(frame)

            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

            # make a prediction on the ROI, then lookup the class

                preds = emotion_model.predict(roi)[0]
                label = emotion_label[preds.argmax()]

            # We are starting the clock here and after every 10 seconds 
            # we will give a voice prediction.

                start = time.clock()
                print(start)

                if int(start) % 10 == 0:
                    if label == "Angry":
                        playsound("reactions/anger.mp3")
                    elif label == "Disgust":
                        playsound("reactions/disgust.mp3")
                    elif label == "Fear":
                        playsound("reactions/fear.mp3")

                    elif label == "Happy":
                        playsound("reactions/happy.mp3")

                    if label == "Neutral":
                        playsound("reactions/neutral.mp3")

                    elif label == "Sad":
                        playsound("reactions/sad.mp3")

                    elif label == "Surprise":
                        playsound("reactions/surprise.mp3")

                label_position = (x,y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

            else:
                cv2.putText(frame, 'No Face Found', (20,60), cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

        cv2.imshow('Emotion Detector',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


else:
    print("Enter 1 to begin")