import imutils
from imutils.video import VideoStream
import numpy as np
import cv2
import os
import time


def run(student_id):
    # Loading face detector
    protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
    modelPath = os.path.sep.join(["face_detection_model", "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    inputId = str(student_id)
    dataPath = "dataSet/" + inputId
    if not os.path.isdir(dataPath):
        os.makedirs(dataPath)
    sampleNum = 0

    # Starting video stream
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    while True:
        frame = vs.read()
        savedFrame = frame
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(imageBlob)
        detections = detector.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue

                sampleNum += 1

                cv2.imwrite(dataPath + "/" + inputId + "_" + str(sampleNum) + ".jpg", savedFrame)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.waitKey(100)

        cv2.imshow("Adding new face data", frame)
        cv2.waitKey(1)
        if sampleNum >= 30:
            break

    cv2.destroyAllWindows()
    vs.stop()
