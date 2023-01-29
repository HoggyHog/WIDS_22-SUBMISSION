# WIDS_22-SUBMISSION

## INSTALLATIONS TO PERFORM INITIALLY

1) pip install gtts
2) pip install playsound
3) pip install tensorflow

Run the files in the following order
1) reaction.py
2) train.py
3) final_run.py

## CODE RUNTHROUGH

#reaction.py

Basically creates a folder 'reactions' to store voice recordings for each sort of emotion such gtts and playsound

#train.py

1) First we read the data here from the 'fer2013.csv' dataset which is available online. After that, the images are formed and split into folders names on the emotion and also into if they are for training and testing
2) After this, the images are now augmented using ImageDataGenerator from keras
3) Then the model is framed with the architecture of 3 Convolutional Blocks (2 Conv2D layers immediately followed by 2 BatchNormalizations, and then MaxPooling followed by Dropout. Then there's a Flatten and Dense to fully connect the model
4) As for the last layer, it will be a Dense layer with a softmax activation function.
5) Now the model is trained and after this the best weights are stored in emotions.h5 file, so that it can be used in the later files

#final_run.py

1) So first the model is loaded from 'emotions.h5'
2) Next using the webcam, a photo is taken on which the face is detected using a haarcascade, which is then converted to gray scale.
3) Now this image is passed into the model for predicting the emotion, and for every 10 seconds, plays a sound on the emotion detected


