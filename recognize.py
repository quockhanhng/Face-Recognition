from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import mysql.connector
from datetime import datetime
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.8, help="minimum probability to filter weak detections")
ap.add_argument("-i", "--class_id", required=True, help="determine which class need to check in")
args = vars(ap.parse_args())

print("[LOG] Loading Face Detector")
protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
modelPath = os.path.sep.join(["face_detection_model", "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print("[LOG] Loading Face Embedding Model")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# Load the face recognition model with the label encoder
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())

col_names = ['Student_Id', 'Subject_Id', 'Date_Time']
attendance = pd.DataFrame(columns=col_names)


# get list of student from class with subject's id


def getStudentsFromClass(class_id):
    my_db = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="ThisIsPassword",
        database="face_recognition"
    )
    my_cursor = my_db.cursor()
    query = """
            SELECT DISTINCT u.code
            FROM users as u
            INNER JOIN user_subject as us
            ON u.id = us.user_id 
            INNER JOIN subjects as s
            ON us.subject_id = s.id WHERE s.code = '{}'
            AND u.role = 2""".format(class_id)

    my_cursor.execute(query)
    my_result = [item[0] for item in my_cursor.fetchall()]
    my_cursor.close()

    return my_result


def checkInStudent(s_code, s_id, c_id, attendance_time):
    my_db = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="ThisIsPassword",
        database="face_recognition"
    )
    query = """
            INSERT INTO recognitions (user_id, subject_id, created_at, updated_at) 
            VALUES ('{}', '{}', '{}', '{}')
            """.format(s_id, c_id, attendance_time, attendance_time)
    my_cursor = my_db.cursor()
    my_cursor.execute(query)
    my_db.commit()
    print("[LOG] Check in student {} successful".format(s_code))
    my_cursor.close()


def getStudentIdFromCode(code):
    my_db = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="ThisIsPassword",
        database="face_recognition"
    )
    my_cursor = my_db.cursor()
    query = """
            SELECT u.id
            FROM users as u
            WHERE u.code = '{}'""".format(code)

    my_cursor.execute(query)
    my_result = [item[0] for item in my_cursor.fetchall()]
    my_cursor.close()

    return my_result[0]


def getSubjectIdFromCode(code):
    my_db = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="ThisIsPassword",
        database="face_recognition"
    )
    my_cursor = my_db.cursor()
    query = """
            SELECT s.id
            FROM subjects as s
            WHERE s.code = '{}'""".format(code)

    my_cursor.execute(query)
    my_result = [item[0] for item in my_cursor.fetchall()]
    my_cursor.close()

    return my_result[0]


def getEmailFromStudentId(code):
    my_db = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="ThisIsPassword",
        database="face_recognition"
    )
    my_cursor = my_db.cursor()
    query = """
                SELECT u.email
                FROM users as u
                WHERE u.code = '{}'""".format(code)

    my_cursor.execute(query)
    my_result = [item[0] for item in my_cursor.fetchall()]
    my_cursor.close()

    return my_result[0]


def sendMail(receiver, body):
    command = "python send_mail.py --receive \"{}\" --body \"{}\"".format(receiver, body)
    os.system(command)


def sendMailWithAttachment(receiver, body, file):
    command = "python send_mail.py --receive \"{}\" --body \"{}\" --file \"{}\"".format(receiver, body, file)
    os.system(command)


subject_code = str(args["class_id"])
student_list = getStudentsFromClass(subject_code)
student_check_in_list = {i: False for i in student_list}

print("[LOG] starting video stream")
vs = VideoStream(src=0).start()
time.sleep(1.0)

while True:
    frame = vs.read()

    # Resize frame image to 600 px width then grab the image dimensions
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # Construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # Apply OpenCV's deep learning-based face detector to localize faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Ensure confidence satisfy minimum threshold
        if confidence > args["confidence"]:
            # Compute (x, y) coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:  # Confirm the face's width and height are not too small
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            student_code = le.classes_[j]

            # Check if student is in class
            # text = "{}: {:.2f}%".format(student_code, proba * 100)
            text = "Not in this class"
            if student_code in student_list and proba > 0.65:
                text = "{}: {:.2f}%".format(student_code, proba * 100)
                if not student_check_in_list[student_code]:
                    student_check_in_list[student_code] = True
                    student_id = getStudentIdFromCode(student_code)
                    subject_id = getSubjectIdFromCode(subject_code)

                    now = datetime.now()
                    current_time = now.strftime("%Y-%m-%d %H:%M:%S")

                    checkInStudent(student_code, student_id, subject_id, current_time)
                    attendance.loc[len(attendance)] = [student_id, subject_id, current_time]

            # Draw the bounding box and probability of the face
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):  # Stop when press 'Q'
        break

# Write attendance log for this lesson
now = datetime.now()
current_time = now.strftime("%Y-%m-%d_%H-%M-%S")

fileName = "C:\\Users\\ADMIN\\PycharmProjects\\Face Recognition\\attendance"
fileName = fileName + os.sep + "Attendance_" + current_time + ".csv"
attendance.to_csv(fileName, index=False)

# Send warning mail to absent students
for student_code in student_check_in_list:
    if not student_check_in_list[student_code]:
        sendMailWithAttachment(getEmailFromStudentId(student_code),
                               "Bạn đã vắng mặt ở tiết học {}".format(subject_code),
                               fileName)

cv2.destroyAllWindows()
vs.stop()
