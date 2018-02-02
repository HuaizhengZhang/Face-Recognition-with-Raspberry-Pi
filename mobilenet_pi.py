import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
import time
from picamera.array import PiRGBArray
from picamera import PiCamera

model = MobileNet(weights='imagenet')
print "Load model successfully"


camera = PiCamera()
camera.resolution = (600,600)
camera.framerate = 32
cap = PiRGBArray(camera, size=(600, 600))
print "Start video stream"


for i in camera.capture_continuous(cap, format='bgr', use_video_port=True):
    frame = i.array
   
    image = cv2.resize(frame, (224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    preds = model.predict(img)
    
    
    cv2.putText(frame, 'Predicted: ' + str(decode_predictions(preds, top=1)[0]), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    cap.truncate(0)
    if key == ord('q'):
        break
