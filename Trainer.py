import cv2
import os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
dataPath = 'dataSet'


def getImageWithId(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]  # Image training path
    faceArray = []
    IdArray = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        # split to get ID of the image
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faceArray.append(faceNp)
        print(ID)
        IdArray.append(ID)
        cv2.imshow("training", faceNp)
        cv2.waitKey(10)
    return np.array(IdArray), faceArray


Ids, faces = getImageWithId(dataPath)
recognizer.train(faces, Ids)
recognizer.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()
