import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('model_v6_23.hdf5')

EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

def detect_emotion(face_img):
    face_img = cv2.resize(face_img, (48, 48))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = face_img.astype("float") / 255.0
    face_img = np.reshape(face_img, [1, face_img.shape[0], face_img.shape[1], 1])

    predictions = model.predict(face_img)
    return EMOTIONS[np.argmax(predictions)]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        emotion = detect_emotion(face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
