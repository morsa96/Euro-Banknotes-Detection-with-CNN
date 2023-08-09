import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def class_predicted(prediction):
    pred = np.argmax(prediction[0])
    return (CATEGORIES[pred])

def prepare(img_path):
    img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
    x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
    x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)
    # Rescale by 1/255
    x /= 255
    return x

def play_video(video_path):
    count=0
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            # Release the Video if ret is false
            cap.release()
            print("Released Video Resource")
            break
        cv2.imwrite("frame%d.jpg"% count,frame)
        prediction = model.predict(prepare("frame%d.jpg"% count))
        pred = class_predicted(prediction)
        cv2.putText(
            frame, #numpy array on which text is written
            pred, #text
            position, #position at which writing has to start
            cv2.FONT_HERSHEY_SIMPLEX, #font family
            1, #font size
            (209, 80, 0, 255), #font color
            3) #font stroke"""
        # Displaying with OpenCV
        cv2.imshow('frame', frame)
        os.remove("frame%d.jpg"% count)
        count+=1
        # Stop playing when entered 'q' from keyboard
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

position = (10,50)
    
model = keras.models.load_model("euro-CNN.h5")
CATEGORIES = ["5-euro","10-euro","20-euro","50-euro"]

#test with a video
play_video('prova.mov')



