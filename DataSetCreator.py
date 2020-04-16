import cv2
import sqlite3
from pip._vendor.distlib.compat import raw_input

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def insertOrUpdate(student_id, student_name, student_class, student_gender):
    conn = sqlite3.connect("StudentDatabase.db")
    commandString = "SELECT * FROM Student WHERE ID = " + str(student_id)
    cursor = conn.execute(commandString)
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
    if isRecordExist == 1:
        commandString = "UPDATE Student SET Name = " + str(student_name) + ", Class =" + \
                        str(student_class) + ", Gender = " + str(student_gender) + " WHERE ID = " + str(student_id)
    else:
        commandString = "INSERT INTO Student(ID,Name,Class,Gender) VALUES(" + str(student_id) + ", " + \
                        str(student_name) + ", " + str(student_class) + "," + str(student_gender) + ")"
    conn.execute(commandString)
    conn.commit()
    conn.close()


inputId = raw_input("Enter student's id: ")
inputName = raw_input("Enter name: ")
inputClass = raw_input("Enter class: ")
inputGender = raw_input("Enter gender: ")
insertOrUpdate(inputId, inputName, inputClass, inputGender)
sampleNum = 0
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        sampleNum += 1
        cv2.imwrite("dataSet/User." + str(inputId) + "." + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.waitKey(100)
    cv2.imshow("Face", img)
    cv2.waitKey(1)
    if sampleNum > 20:
        break
cam.release()
cv2.destroyAllWindows()
