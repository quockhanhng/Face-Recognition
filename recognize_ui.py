import tkinter
import tkinter.scrolledtext as tkst
import cv2
import PIL.Image
import PIL.ImageTk
import imutils
from imutils import paths
from imutils.video import VideoStream
import time
from datetime import datetime
import os
import pickle
import numpy as np
import pandas as pd
import mysql.connector

import send_mail


class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.window.resizable(False, False)
        self.window.geometry("1000x490+450+250")

        self.is_recognize = False
        self.subject_code = ""
        self.student_list = []
        self.student_check_in_list = []
        self.col_names = ['Student_Id', 'Subject_Id', 'Date_Time']
        self.attendance = pd.DataFrame(columns=self.col_names)

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create the whole window panel
        self.m = tkinter.PanedWindow()
        self.m.pack(fill=tkinter.BOTH, expand=1)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.m.add(self.canvas)

        # Interaction panel
        self.m2 = tkinter.PanedWindow(self.m, orient=tkinter.VERTICAL)
        self.m.add(self.m2)

        # Title
        self.lb_title = tkinter.Label(self.m2, text="Attendance System", font='Monospaced 12 bold')
        self.lb_title.config(background='#446dc7', height=3)
        self.m2.add(self.lb_title)
        # Input class' code
        self.m3 = tkinter.PanedWindow(self.m2, orient=tkinter.HORIZONTAL)
        self.m2.add(self.m3)
        self.lb_text = tkinter.Label(self.m3, text="Enter Class' Code:", width=25, font='Monospaced 10')
        self.lb_text.config(height=2)
        self.m3.add(self.lb_text)
        self.entry_class_code = tkinter.Entry(self.m3, width=25, borderwidth=5, font='Monospaced 10')
        self.m3.add(self.entry_class_code)
        # Start button
        self.btn_start = \
            tkinter.Button(self.m2, text="Start check-in students", width=50, borderwidth=2
                           , font='Monospaced 10 bold', command=self.onStart)
        self.btn_start.config(height=3)
        self.m2.add(self.btn_start)
        # Student's id
        self.lb_student_code = tkinter.Label(self.m2, text="", width=50, font='Monospaced 10')
        self.lb_student_code.config(background='#446dc7', height=3)
        self.m2.add(self.lb_student_code)
        # Scroll log
        self.log_text = tkst.ScrolledText(master=self.m2, wrap=tkinter.WORD, width=50, height=13)
        self.m2.add(self.log_text)
        # Stop button
        self.btn_stop = \
            tkinter.Button(self.m2, text="STOP", width=48, borderwidth=2,
                           font='Monospaced 10 bold', command=self.onStop)
        self.btn_stop.config(height=3)
        self.m2.add(self.btn_stop)

        self.checked_students = {}
        self.checked_student_count = {}
        self.photo_change_student = {}

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()
        self.window.mainloop()

    def onStart(self):
        if not self.is_recognize:

            if checkClassIfExits(self.entry_class_code.get()):
                self.is_recognize = True

                self.lb_student_code['text'] = ""
                self.lb_student_code.config(background='#446dc7', height=3)

                code = self.entry_class_code.get()
                self.subject_code = code

                self.log_text.insert(tkinter.INSERT, "[LOG] Start check-in students in class {}\n".format(code))

                self.student_list = getStudentsFromClass(code)
                self.student_check_in_list = {i: False for i in self.student_list}

                # Setup checkboxes to check in students
                self.checkboxes_panel = VerticalScrolledFrame(self.m)
                self.checkboxes_list = dict()
                for s_id in self.student_list:
                    self.addCheckbox(s_id)

                self.m.add(self.checkboxes_panel)
                self.window.geometry("1200x490+450+250")

            else:
                self.lb_student_code['text'] = "Could not find class"
                self.lb_student_code.config(background='#EB3912', height=3)

        else:
            self.log_text.insert(tkinter.INSERT, "[LOG] Finish this session first in order to start another session\n")

    def onStop(self):
        if self.is_recognize:
            self.is_recognize = False
            self.checked_student_count = {}

            self.log_text.insert(tkinter.INSERT, "[LOG] Finished check-in\n")

            self.lb_student_code['text'] = ""
            self.lb_student_code.config(background='#446dc7', height=3)
            self.entry_class_code.delete(0, 'end')

            self.window.geometry("1000x490+450+250")
            self.checkboxes_panel.destroy()

            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d_%H-%M-%S")

            # Actually check-in student in database
            for code in self.checked_students:
                student_id = getStudentIdFromCode(code)
                subject_id = getSubjectIdFromCode(self.subject_code)
                self.attendance.loc[len(self.attendance)] = [student_id, subject_id, current_time]
                checkInStudent(student_id, subject_id, current_time)

            # Write attendance log for this lesson
            fileName = "C:\\Users\\ADMIN\\PycharmProjects\\Face Recognition\\attendance"
            fileName = fileName + os.sep + "Attendance_" + current_time + ".csv"
            self.attendance.to_csv(fileName, index=False)
            self.log_text.insert(tkinter.INSERT, "[LOG] Wrote log for this lesson to file successful\n")

            # Update student's data set
            for code in self.photo_change_student:
                imagePaths = list(paths.list_images("dataSet/{}".format(code)))
                for (i, _) in enumerate(imagePaths):
                    imagePath = "dataSet/{}/{}_{}.jpg".format(code, code, i + 1)
                    newImagePath = "dataSet/{}/{}_{}.jpg".format(code, code, i)
                    # num = imagePath.split(os.path.sep)[-1].split("_")[1]

                    os.rename(r'{}'.format(imagePath), r'{}'.format(newImagePath))
                newImage = "dataSet/{}/{}_{}.jpg".format(code, code, len(imagePaths))
                cv2.imwrite(newImage, self.photo_change_student[code])
                removeImage = "dataSet/{}/{}_0.jpg".format(code, code)
                os.remove(removeImage)

            # # Send warning mail to absent students
            # for student_code in self.student_check_in_list:
            #     if not self.student_check_in_list[student_code]:
            #         sendMail(getEmailFromId(student_code),
            #                  "Bạn đã vắng mặt ở tiết học {}".format(self.subject_code))
            #
            # # Send report to lecturer
            # lecturer_list = getLecturersFromClass(self.subject_code)
            # for lecturer_code in lecturer_list:
            #     sendMailWithAttachment(getEmailFromId(lecturer_code), "Attendance Report", fileName)

        else:
            self.log_text.insert(tkinter.INSERT, "[LOG] The attendance system is not start yet\n")

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            # Grab the image dimensions
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
                if confidence > 0.8:
                    # Compute (x, y) coordinates of the bounding box
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Extract the face ROI
                    face = frame[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]

                    if fW < 20 or fH < 20:  # Confirm the face's width and height are not too small
                        continue

                    # Draw the bounding box and probability of the face
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        if self.is_recognize:
            self.window.after(self.delay, self.process)
        else:
            self.window.after(self.delay, self.update)

    def process(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            # Grab the image dimensions
            savedFrame = frame
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
                if confidence > 0.8:
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
                    if student_code in self.checked_student_count:
                        self.checked_student_count['{}'.format(student_code)] += 1
                    else:
                        self.checked_student_count['{}'.format(student_code)] = 1
                    print(student_code + " " + str(proba))
                    # Check if student is in class
                    text = "Not in this class"
                    if student_code in self.student_list and proba > 0.5 \
                            and self.checked_student_count['{}'.format(student_code)] > 5:
                        text = "{}: {:.2f}%".format(student_code, proba * 100)
                        if not self.student_check_in_list[student_code]:
                            self.student_check_in_list[student_code] = True

                            now = datetime.now()
                            current_time = now.strftime("%Y-%m-%d %H:%M:%S")

                            # Add student to checked list
                            self.checked_students['{}'.format(student_code)] = current_time
                            self.checkboxes_list[student_code].var.set(1)

                            # Write face log to local storage
                            check_in_time = current_time.split(" ")[1].split(":")
                            dataPath = "attendanceFaceLog/" + self.entry_class_code.get() \
                                       + "/" + current_time.split(" ")[0]
                            if not os.path.isdir(dataPath):
                                os.makedirs(dataPath)
                            cv2.imwrite(dataPath + "/" + student_code + "_" + current_time.split(" ")[0] + "_" +
                                        check_in_time[0] + "-" + check_in_time[1] + "-" + check_in_time[2] + ".jpg",
                                        savedFrame)

                            self.photo_change_student['{}'.format(student_code)] = savedFrame

                            log_text = "Check in student {} successful".format(student_code)
                            self.lb_student_code['text'] = log_text
                            self.lb_student_code.config(background='#4cc845', height=3)
                            self.log_text.insert(tkinter.INSERT, "[LOG] " + log_text + "\n")

                    # Draw the bounding box and probability of the face
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        if self.is_recognize:
            self.window.after(self.delay, self.process)
        else:
            self.window.after(self.delay, self.update)

    def addCheckbox(self, s_id):
        self.checkboxes_list[s_id] = tkinter.Checkbutton(self.checkboxes_panel.interior, text=s_id)
        self.checkboxes_list[s_id].var = tkinter.IntVar()
        self.checkboxes_list[s_id]['variable'] = self.checkboxes_list[s_id].var
        self.checkboxes_list[s_id]['command'] = lambda w=self.checkboxes_list[s_id]: self.upon_select(w)
        self.checkboxes_list[s_id].pack(padx=10, pady=5, side=tkinter.TOP)

    def upon_select(self, widget):
        if widget.var.get() == 0:  # Lecturer uncheck student
            self.student_check_in_list[widget['text']] = False
            self.checked_student_count['{}'.format(widget['text'])] = 0
            log_text = "Unchecked student {}".format(widget['text'])
            self.lb_student_code['text'] = log_text
            self.lb_student_code.config(background='#BB0A1E', height=3)
            self.log_text.insert(tkinter.INSERT, "[LOG] " + log_text + "\n")
            self.checked_students.pop(widget['text'])
        else:  # Lecturer check student manually
            self.student_check_in_list[widget['text']] = True
            log_text = "Check in student {} successful".format(widget['text'])
            self.lb_student_code['text'] = log_text
            self.lb_student_code.config(background='#4cc845', height=3)
            self.log_text.insert(tkinter.INSERT, "[LOG] " + log_text + "\n")
            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d %H:%M:%S")
            self.checked_students['{}'.format(widget['text'])] = current_time


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return ret, None

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


class VerticalScrolledFrame(tkinter.Frame):
    def __init__(self, parent, *args, **kw):
        tkinter.Frame.__init__(self, parent, *args, **kw)

        # create a canvas object and a vertical scrollbar for scrolling it
        v_scrollbar = tkinter.Scrollbar(self, orient=tkinter.VERTICAL)
        v_scrollbar.pack(fill=tkinter.Y, side=tkinter.RIGHT, expand=tkinter.FALSE)
        canvas = tkinter.Canvas(self, bd=0, highlightthickness=0,
                                yscrollcommand=v_scrollbar.set)
        canvas.pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=tkinter.TRUE)
        v_scrollbar.config(command=canvas.yview)

        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.interior = interior = tkinter.Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior,
                                           anchor=tkinter.NW)

        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.config(width=interior.winfo_reqwidth())

        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())

        canvas.bind('<Configure>', _configure_canvas)


# Loading Face Detector
protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
modelPath = os.path.sep.join(["face_detection_model", "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Loading Face Embedding Model
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# Load the face recognition model with the label encoder
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())


def checkClassIfExits(code):
    my_db = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="ThisIsPassword",
        database="face_recognition"
    )
    my_cursor = my_db.cursor()
    query = """
            SELECT COUNT(s.id)
            FROM subjects as s
            WHERE s.code = '{}'""".format(code)

    my_cursor.execute(query)
    my_result = my_cursor.fetchone()[0]
    my_cursor.close()

    return False if my_result == 0 else True


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


def getLecturersFromClass(class_id):
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
            AND u.role = 1""".format(class_id)

    my_cursor.execute(query)
    my_result = [item[0] for item in my_cursor.fetchall()]
    my_cursor.close()

    return my_result


def checkInStudent(s_id, c_id, attendance_time):
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
    my_result = my_cursor.fetchone()[0]
    my_cursor.close()

    return my_result


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
    my_result = my_cursor.fetchone()[0]
    my_cursor.close()

    return my_result


def getEmailFromId(code):
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
    my_result = my_cursor.fetchone()[0]
    my_cursor.close()

    return my_result


def sendMail(receiver, body):
    send_mail.runWithoutAttachment(receiver, body)


def sendMailWithAttachment(receiver, body, file):
    send_mail.runWithAttachment(receiver, body, file)


# Create a window and pass it to the Application object
def run():
    App(tkinter.Tk(), "Student Attendance System")


run()
