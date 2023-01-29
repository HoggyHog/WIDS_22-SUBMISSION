import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import os

#reading in the data
data=pd.read_csv('fer2013.csv')

#DATA PREPROCESSING
# In this section, we take the image array data and convert them to images of 48x48 pixels using opencv, and store them in their respective folder for training and validation
count0=0
count1=0
count2=0
count3=0
count4=0
count5=0
count6=0


parent_path="C:/Users/karth/OneDrive/Desktop/CODES/ML/WIDS/"

for i,j,k in tqdm(zip(data['emotion'],data['pixels'],data['Usage'])):
    pixel=[]
    pixels=j.split(' ')
    for i in pixels:
        i=float(i)
        pixel.append(i)
    pixel=np.array(pixel)
    pixel=pixel.reshape(48,48)
    
    if k=='Training':
        type="train"
        if not os.path.exists(os.path.join(parent_path,type)):
            os.mkdir(os.path.join(parent_path,type))
        if i==0:
            emotion="angry"
            path_check=os.path.join(parent_path,type,emotion)
            if not os.path.exists(path_check):
                os.mkdir(path_check)
            file_save=path_check+"/"+str(count0)+".jpg"
            cv2.imwrite(file_save,pixel)
            count0+=1

        if i==1:
            emotion="disgust"
            path_check=os.path.join(parent_path,type,emotion)
            if not os.path.exists(path_check):
                os.mkdir(path_check)
            file_save=path_check+"/"+str(count1)+".jpg"
            cv2.imwrite(file_save,pixel)
            count1+=1
        
        if i==2:
            emotion="fear"
            path_check=os.path.join(parent_path,type,emotion)
            if not os.path.exists(path_check):
                os.mkdir(path_check)
            file_save=path_check+"/"+str(count2)+".jpg"
            cv2.imwrite(file_save,pixel)
            count2+=1
        
        if i==3:
            emotion="happy"
            path_check=os.path.join(parent_path,type,emotion)
            if not os.path.exists(path_check):
                os.mkdir(path_check)
            file_save=path_check+"/"+str(count3)+".jpg"
            cv2.imwrite(file_save,pixel)
            count3+=1
        
        if i==4:
            emotion="sad"
            path_check=os.path.join(parent_path,type,emotion)
            if not os.path.exists(path_check):
                os.mkdir(path_check)
            file_save=path_check+"/"+str(count4)+".jpg"
            cv2.imwrite(file_save,pixel)
            count4+=1
        
        if i==5:
            emotion="surprise"
            path_check=os.path.join(parent_path,type,emotion)
            if not os.path.exists(path_check):
                os.mkdir(path_check)
            file_save=path_check+"/"+str(count5)+".jpg"
            cv2.imwrite(file_save,pixel)
            count5+=1
        
        if i==6:
            emotion="neutral"
            path_check=os.path.join(parent_path,type,emotion)
            if not os.path.exists(path_check):
                os.mkdir(path_check)
            file_save=path_check+"/"+str(count6)+".jpg"
            cv2.imwrite(file_save,pixel)
            count6+=1


    if k=='PublicTest':
        type="pub_test"
        if not os.path.exists(os.path.join(parent_path,type)):
            os.mkdir(os.path.join(parent_path,type))
        if i==0:
            emotion="angry"
            path_check=os.path.join(parent_path,type,emotion)
            if not os.path.exists(path_check):
                os.mkdir(path_check)
            file_save=path_check+"/"+str(count0)+".jpg"
            cv2.imwrite(file_save,pixel)
            count0+=1

        if i==1:
            emotion="disgust"
            path_check=os.path.join(parent_path,type,emotion)
            if not os.path.exists(path_check):
                os.mkdir(path_check)
            file_save=path_check+"/"+str(count1)+".jpg"
            cv2.imwrite(file_save,pixel)
            count1+=1
        
        if i==2:
            emotion="fear"
            path_check=os.path.join(parent_path,type,emotion)
            if not os.path.exists(path_check):
                os.mkdir(path_check)
            file_save=path_check+"/"+str(count2)+".jpg"
            cv2.imwrite(file_save,pixel)
            count2+=1
        if i==3:
            emotion="happy"
            path_check=os.path.join(parent_path,type,emotion)
            if not os.path.exists(path_check):
                os.mkdir(path_check)
            file_save=path_check+"/"+str(count3)+".jpg"
            cv2.imwrite(file_save,pixel)
            count3+=1
        
        if i==4:
            emotion="sad"
            path_check=os.path.join(parent_path,type,emotion)
            if not os.path.exists(path_check):
                os.mkdir(path_check)
            file_save=path_check+"/"+str(count4)+".jpg"
            cv2.imwrite(file_save,pixel)
            count4+=1
        
        if i==5:
            emotion="surprise"
            path_check=os.path.join(parent_path,type,emotion)
            if not os.path.exists(path_check):
                os.mkdir(path_check)
            file_save=path_check+"/"+str(count5)+".jpg"
            cv2.imwrite(file_save,pixel)
            count5+=1
        
        if i==6:
            emotion="neutral"
            path_check=os.path.join(parent_path,type,emotion)
            if not os.path.exists(path_check):
                os.mkdir(path_check)
            file_save=path_check+"/"+str(count6)+".jpg"
            cv2.imwrite(file_save,pixel)
            count6+=1

    if k=='PrivateTest':
        type="pri_test"
        if not os.path.exists(os.path.join(parent_path,type)):
            os.mkdir(os.path.join(parent_path,type))
        if i==0:
            emotion="angry"
            path_check=os.path.join(parent_path,type,emotion)
            if not os.path.exists(path_check):
                os.mkdir(path_check)
            file_save=path_check+"/"+str(count0)+".jpg"
            cv2.imwrite(file_save,pixel)
            count0+=1

        if i==1:
            emotion="disgust"
            path_check=os.path.join(parent_path,type,emotion)
            if not os.path.exists(path_check):
                os.mkdir(path_check)
            file_save=path_check+"/"+str(count1)+".jpg"
            cv2.imwrite(file_save,pixel)
            count1+=1
        
        if i==2:
            emotion="fear"
            path_check=os.path.join(parent_path,type,emotion)
            if not os.path.exists(path_check):
                os.mkdir(path_check)
            file_save=path_check+"/"+str(count2)+".jpg"
            cv2.imwrite(file_save,pixel)
            count2+=1
        if i==3:
            emotion="happy"
            path_check=os.path.join(parent_path,type,emotion)
            if not os.path.exists(path_check):
                os.mkdir(path_check)
            file_save=path_check+"/"+str(count3)+".jpg"
            cv2.imwrite(file_save,pixel)
            count3+=1
        
        if i==4:
            emotion="sad"
            path_check=os.path.join(parent_path,type,emotion)
            if not os.path.exists(path_check):
                os.mkdir(path_check)
            file_save=path_check+"/"+str(count4)+".jpg"
            cv2.imwrite(file_save,pixel)
            count4+=1
        
        if i==5:
            emotion="surprise"
            path_check=os.path.join(parent_path,type,emotion)
            if not os.path.exists(path_check):
                os.mkdir(path_check)
            file_save=path_check+"/"+str(count5)+".jpg"
            cv2.imwrite(file_save,pixel)
            count5+=1
        
       
        if i==6:
            emotion="neutral"
            path_check=os.path.join(parent_path,type,emotion)
            if not os.path.exists(path_check):
                os.mkdir(path_check)
            file_save=path_check+"/"+str(count6)+".jpg"
            cv2.imwrite(file_save,pixel)
            count6+=1


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, Flatten, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

num_classes = 7

Img_Height = 48
Img_width = 48

batch_size = 32
train_dir = "./train"
validation_dir = "./pri_test"

#Images are now augmented to train the model better

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=60,
                                   shear_range=0.5,
                                   zoom_range=0.5,
                                   width_shift_range=0.5,
                                   height_shift_range=0.5,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    color_mode='grayscale',
                                                    target_size=(Img_Height, Img_width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              color_mode='grayscale',
                                                              target_size=(Img_Height, Img_width),
                                                              batch_size=batch_size,
                                                              class_mode='categorical',
                                                              shuffle=True)


#In this section of the code, we define the Model and all the Layers inside of it

model = Sequential()

# Block-1: The First Convolutional Block

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', 
                 kernel_initializer='he_normal',
                 activation="elu", 
                 input_shape=(Img_Height, Img_width, 1), 
                 name="Conv1"))

model.add(BatchNormalization(name="Batch_Norm1"))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', 
                 kernel_initializer='he_normal', 
                 activation="elu", name="Conv2"))

model.add(BatchNormalization(name="Batch_Norm2"))
model.add(MaxPooling2D(pool_size=(2,2), name="Maxpool1"))
model.add(Dropout(0.2, name="Dropout1"))

# Block-2: The Second Convolutional Block

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', 
                 kernel_initializer='he_normal',
                 activation="elu", name="Conv3"))

model.add(BatchNormalization(name="Batch_Norm3"))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',
                 kernel_initializer='he_normal', 
                 activation="elu", name="Conv4"))

model.add(BatchNormalization(name="Batch_Norm4"))
model.add(MaxPooling2D(pool_size=(2,2), name="Maxpool2"))
model.add(Dropout(0.2, name="Dropout2"))

# Block-3: The Third Convolutional Block

model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', 
                 kernel_initializer='he_normal', 
                 activation="elu", name="Conv5"))

model.add(BatchNormalization(name="Batch_Norm5"))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', 
                 kernel_initializer='he_normal',
                 activation="elu", name="Conv6"))

model.add(BatchNormalization(name="Batch_Norm6"))
model.add(MaxPooling2D(pool_size=(2,2), name="Maxpool3"))
model.add(Dropout(0.2, name="Dropout3"))

# Block-4: The Fully Connected Block

model.add(Flatten(name="Flatten"))
model.add(Dense(64, activation="elu", kernel_initializer='he_normal', name="Dense"))
model.add(BatchNormalization(name="Batch_Norm7"))
model.add(Dropout(0.5, name="Dropout4"))

# Block-5: The Output Block

model.add(Dense(num_classes, activation="softmax", kernel_initializer='he_normal', name = "Output"))


#importing in Callbacks

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard

checkpoint = ModelCheckpoint("emotions.h5", monitor='accuracy', verbose=1,
                              save_best_only=True, mode='auto', period=1)
#this callback only saves the best weights into emotion.h5 file, which will be used after training

reduce = ReduceLROnPlateau(monitor='accuracy', factor=0.2, patience=10, 
                           min_lr=0.0001, verbose = 1)
#this callback reduces the LR if theres no change in accuracy after 10 epochs


logdir='logs'
tensorboard_Visualization = TensorBoard(log_dir=logdir, histogram_freq=False)
#this callback just plots a graph of the loss function and the accuracy

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr = 0.001),
              metrics=['accuracy'])


train_samples = 28353
validation_samples = 3534
epochs = 150
batch_size = 64

#now we fit the model

model.fit(train_generator,
          #steps_per_epoch = train_samples//batch_size,
          epochs = epochs,
          callbacks = [checkpoint, reduce, tensorboard_Visualization],
          validation_data = validation_generator,
          #validation_steps = validation_samples//batch_size,
          shuffle=True)

