import cv2
import pickle

face_cascade = cv2.CascadeClassifier('CSCD/data/haarcascade_frontalface_alt2.xml')
recog = cv2.face.LBPHFaceRecognizer_create()
recog.read("train.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, a, b) in faces:
        #print(x, y, w, h)
        roi_gray = gray[y:y+b, x:x+a]
        roi_color = frame[y:y+b, x:x+a]

        id_, conf = recog.predict(roi_gray)
        if conf >= 45:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_DUPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

        color = (0, 0, 255)
        stroke = 2
        end_cord_x = x + a
        end_cord_y = y + b
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    cv2.imshow("frame", frame)
    if cv2.waitKey(20) and 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
