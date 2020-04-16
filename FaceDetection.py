import cv2
import sqlite3

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer\\trainingData.yml")
userId = 0
fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL


def getStudentProfile(student_id):
    conn = sqlite3.connect("StudentDatabase.db")
    commandString = "SELECT * FROM Student WHERE ID = " + str(student_id)
    cursor = conn.execute(commandString)
    student_profile = None
    for row in cursor:
        student_profile = row
    conn.commit()
    conn.close()
    return student_profile


while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        userId, conf = rec.predict(gray[y:y + h, x:x + w])
        print(str(userId) + " " + str(conf))
        if conf > 50:
            userId = -1
        profile = getStudentProfile(userId)
        if profile is None:
            cv2.putText(img, "Unknown", (x, y + h), fontFace, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, str(profile[1]), (x, y + h), fontFace, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, str(profile[0]), (x, y + h + 30), fontFace, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, str(profile[2]), (x, y + h + 60), fontFace, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Face", img)
    if cv2.waitKey(1) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
