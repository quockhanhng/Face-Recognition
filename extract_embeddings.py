from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os


def run():
    print("[LOG] Loading Face Detector")
    protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
    modelPath = os.path.sep.join(["face_detection_model", "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # Loading Face Embedding Model
    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

    print("[LOG] Get Images' path in dataSet")
    imagePaths = list(paths.list_images("dataSet"))

    # Initialize lists of extracted facial embeddings
    knownEmbeddings = []
    knownIds = []

    total = 0  # Number of processed faces

    for (i, imagePath) in enumerate(imagePaths):
        print("[LOG] Processing Image {}/{}".format(i + 1, len(imagePaths)))
        studentId = imagePath.split(os.path.sep)[-2]  # Get student's id from path

        # Load, resize image to 600 px width then grab the image dimensions
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        # Construct blob from image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # Apply OpenCV's deep learning-based face detector to localize faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # Ensure at least one face was found
        if len(detections) > 0:
            # Find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            # Ensure confidence satisfy minimum threshold
            if confidence > 0.7:
                # Compute (x, y) coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Extract the face ROI and grab the ROI dimensions
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:  # Confirm the face's width and height are not too small
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # Add student's name and id to list
                knownIds.append(studentId)
                knownEmbeddings.append(vec.flatten())
                total += 1
            else:
                print("[LOG] Skip this image")

    # Dump the facial embeddings + ids to disk
    print("[LOG] serializing {} encodings".format(total))
    data = {"embeddings": knownEmbeddings, "ids": knownIds}
    f = open("output/embeddings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()
