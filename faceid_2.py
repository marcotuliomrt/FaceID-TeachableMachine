# ------------------------------------------------------------------------------------------------------------
# -- The script performs face DETECTION and RECOGNITION on live webcam image
# -- 06/08/2022 Marco Tulio Masselli Rainho Teixeira
# ------------------------------------------------------------------------------------------------------------


import cv2
import numpy as np
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from keras.models import load_model
from PIL import Image, ImageOps


THRESHOLD = 0.8  # threshhold probability for the detection
TEXT = "<MARCO IDENTIFIED>"

# Load the face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the face recognition model
model = load_model("converted_keras/keras_model.h5")

segmentor = SelfiSegmentation()


cap = cv2.VideoCapture(0)


while True:
 
    _, frame = cap.read()


    # get the faces' bounding box coordinates
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=6, minSize=[60, 60])

    # frame without background
    frame = segmentor.removeBG(frame, (255, 255, 255), threshold=0.6)

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


    for (x, y, w, h) in faces:
        # draw the bounding box on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # face image
        face = frame[y+2:y+h-2, x+2:x+w-2]

        face = cv2.resize(face, (224, 224), interpolation = cv2.INTER_AREA)  # Resized image
        
        # ------ tentaiva de salvar o frame e ler ele com PIL ------
        # cv2.imwrite("data/current.jpg", face) 
        # image = Image.open("data/current.jpg")
        # image_array = np.asarray(image) 



        normalized_image_array = (face.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        data[0] = normalized_image_array
        
        prediction = model.predict(data)
        class_idx = np.argmax(prediction, axis=-1)
        prob = np.amax(prediction)

        if class_idx == 0: # & prob > THRESHOLD:
            cv2.putText(frame, TEXT, (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    cv2.imshow('frame', frame)

    # set the condition of manual brake by pressing "k" on the keyboard  
    if cv2.waitKey(1) & 0xFF==ord("k"):
        break
 
        
# release the webcam 
cap.release()
cv2.destroyAllWindows()