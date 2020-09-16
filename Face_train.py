import os
import cv2
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Image_dir = os.path.join(BASE_DIR, "Image")

face_cascade = cv2.CascadeClassifier('CSCD/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []


for root, dirs, files in os.walk(Image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            # print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # print(label_ids)
            # y_labels.append(label)
            # x_train.append(path)
            pil_image = Image.open(path).convert("L")
            size = (500, 500)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            Image_array = np.array(pil_image, "uint8")
            # print(Image_array)
            faces = face_cascade.detectMultiScale(Image_array, scaleFactor=1.5, minNeighbors=5)

            for(x, y, w, h) in faces:
                roi = Image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)


# print(y_labels)
# print(x_train)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("train.yml")


