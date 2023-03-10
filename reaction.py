"""
    In this python file we will be creating all the required voice
    recordings for both the emotions as well as all the gestures and
    we will be storing them in the reactions directory.

"""

from gtts import gTTS
from playsound import playsound
import shutil
import os

# Set the path to the reactions folder
path = ("reactions")

# Create the reactions directory if it does not exist
if not os.path.exists(path):
    os.mkdir(path)

# Saving our custom reactions

# 1. Emotions Reactions
tts_angry = gTTS("Wow you look cute even when angry!")
tts_angry.save("anger.mp3")

tts_angry = gTTS("Why are you disgusted!")
tts_angry.save("disgust.mp3")

tts_fear = gTTS("Don't be Scared Buddy I will protect you.")
tts_fear.save("fear.mp3")

tts_happy = gTTS("You Look Pretty when you smile!")
tts_happy.save("happy.mp3")

tts_neutral = gTTS("Don't be so serious dude!")
tts_neutral.save("neutral.mp3")

tts_sad = gTTS("Hello There Sad Face I would Appreciate a smile.")
tts_sad.save("sad.mp3")

tts_surprise = gTTS("You seem surprised at how awesome I am.")
tts_surprise.save("surprise.mp3")


files = ['anger.mp3', 'disgust.mp3', 'fear.mp3', 'happy.mp3', 'neutral.mp3', 'sad.mp3', 'surprise.mp3', ] 
         
for f in files:
    shutil.move(f, path)

# test a random sample
#playsound("reactions/neutral.mp3")