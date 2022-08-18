# https://medium.datadriveninvestor.com/multi-class-image-classification-in-teachable-machine-and-its-real-time-detection-with-opencv-282a1409006f

import cv2
import numpy as np
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from keras.models import load_model



TEXT = "<MARCO IDENTIFIED>"


def gen_labels():
        labels = {}
        with open("converted_keras/labels.txt", "r") as label:
            text = label.read()
            lines = text.split("\n")
            print(lines)
            for line in lines[0:-1]:
                    hold = line.split(" ", 1)
                    labels[hold[0]] = hold[1]
        return labels



THRESHOLD = 0.8  # threshhold probability for the detection
TEXT = "<MARCO IDENTIFIED>"

# Load the face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the face recognition model
model = load_model("converted_keras/keras_model.h5")

segmentor = SelfiSegmentation()




# Disable scientific notation for clarity
np.set_printoptions(suppress=True)
cap = cv2.VideoCapture(0)
# Load the model
model = load_model('converted_keras/keras_model.h5')

"""
Create the array of the right shape to feed into the keras model
The 'length' or number of images you can put into the array is
determined by the first position in the shape tuple, in this case 1."""
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# A dict that stores the labels
labels = gen_labels()

while True:
    # Choose a suitable font
    font = cv2.FONT_HERSHEY_SIMPLEX
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    #get the faces' bounding box coordinates
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=6, minSize=[60, 60])
    # frame without background
    #frame = segmentor.removeBG(frame, (255, 255, 255), threshold=0.6)


    for (x, y, w, h) in faces:
        # draw the bounding box on the frame
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # face image
        face = frame[(y+2):(y+h-2), (x+2):(x+w-2)]

        # resize the image to a 224x224 with the same strategy as in TM2:
        # resizing the image to be at least 224x224 and then cropping from the center
        face = cv2.resize(face, (224, 224))
        # turn the image into a numpy array
        image_array = np.asarray(face)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        data[0] = normalized_image_array
        pred = model.predict(data)
        result = np.argmax(pred[0])


        if result == 0: # & prob > THRESHOLD:
            cv2.putText(frame, TEXT, (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


    # set the condition of manual brake by pressing "k" on the keyboard  
    if cv2.waitKey(1) & 0xFF==ord("k"):
        break
    # Show the frame   
    cv2.imshow('Frame', frame)

cap.release()
cv2.destroyAllWindows()