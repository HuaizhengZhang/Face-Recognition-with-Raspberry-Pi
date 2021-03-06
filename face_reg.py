import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import model_from_json
from picamera.arry import PiRGBArray
from picamera import PiCamera
import time

with open('face_model_3.json', 'rb') as json_file:
    model_json = json_file.read()
    
model = model_from_json(model_json)
model.load_weights('face_model_3.h5')
print "Load model successfully"


camera = PiCamera()
camera.resolution = (600,600)
camera.framerate = 32
cap = PiRGBArray(camera, size=(600, 600))
print "Start video stream"

THD = 0.7
for i in camera.capture_continuous(cap, format='bgr', use_video_port=True):
    frame = i.array
    image = cv2.resize(frame, (150, 150))
    image = image.astype('float') / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    predict = model.predict(image)[0]
    
    if predict[0] > THD:
        cv2.putText(frame, "Hello, Dr.Luo Yong", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    elif predict[1] > THD:
        cv2.putText(frame, "Hello, Wang Yongjie", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    elif predict[2] > THD:
        cv2.putText(frame, "Hello, Yi Deliang", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    elif predict[3] > THD:
        cv2.putText(frame, "Hello, Zhang Huaizheng", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    elif predict[4] > THD:
        cv2.putText(frame, "Hello, Dr.Zhou Xin", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Can not recognize", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    cap.truncate(0)
    if key == ord('q'):
        break
