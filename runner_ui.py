from tkinter import *
from tkinter import ttk
from tkinter import messagebox as mb
import tkinter.messagebox
import mysql.connector
import add_face_data
import extract_embeddings
import train_model
import recognize_ui


def check_login(username, password):
    my_db = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="ThisIsPassword",
        database="face_recognition"
    )
    my_cursor = my_db.cursor()
    query = """
                SELECT u.role
                FROM users as u
                WHERE u.code = '{}'
                AND u.plain_password = {}""".format(username, password)

    try:
        my_cursor.execute(query)
        my_result = my_cursor.fetchone()[0]
        my_cursor.close()
        return my_result
    except:
        return -1


def checkStudentIfExits(code):
    my_db = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="ThisIsPassword",
        database="face_recognition"
    )
    my_cursor = my_db.cursor()
    query = """
            SELECT COUNT(u.id)
            FROM users as u
            WHERE u.code = '{}'
            AND u.role = 2""".format(code)

    my_cursor.execute(query)
    my_result = my_cursor.fetchone()[0]
    my_cursor.close()

    return False if my_result == 0 else True


class Authentication:

    def __init__(self, m_root):

        self.root = m_root
        self.root.title('USER AUTHENTICATION')

        rows = 0
        while rows < 10:
            self.root.rowconfigure(rows, weight=1)
            self.root.columnconfigure(rows, weight=1)
            rows += 1

        frame = LabelFrame(self.root, text='Login')
        frame.grid(row=1, column=1, columnspan=10, rowspan=10)

        Label(frame, text=' Username ').grid(row=2, column=1, sticky=W)
        self.username = Entry(frame)
        self.username.grid(row=2, column=2)

        Label(frame, text=' Password ').grid(row=5, column=1, sticky=W)
        self.password = Entry(frame, show='*')
        self.password.grid(row=5, column=2)

        ttk.Button(frame, text='LOGIN', command=self.login_user).grid(row=7, column=2)

        self.message = Label(text='', fg='Red')
        self.message.grid(row=9, column=6)

    def login_user(self):
        input_username = self.username.get()
        input_password = self.password.get()
        result = check_login(input_username, input_password)
        if result == -1:
            self.message['text'] = 'Username or Password incorrect. Try again!'
        elif result == 0:
            self.root.destroy()
            RootUser(tkinter.Tk(), "Root management")
        elif result == 1:
            self.root.destroy()
            recognize_ui.run()
        else:
            self.message['text'] = "You don't have permission to continue"


class RootUser:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.resizable(False, False)
        self.window.geometry("400x200+700+300")

        rows = 0
        while rows < 4:
            self.window.rowconfigure(rows, weight=1)
            self.window.columnconfigure(rows, weight=1)
            rows += 1
        frame = LabelFrame(self.window, text='Management')
        frame.grid(row=1, column=1, columnspan=4, rowspan=4)

        self.entry_student_code = Entry(frame, borderwidth=5, font='Monospaced 10')
        self.entry_student_code.grid(row=2, column=2)

        ttk.Button(frame, text='Add or update face data', command=self.onAddFace).grid(row=2, column=3)
        ttk.Button(frame, text='Train model', command=self.onTrainModel).grid(row=3, column=2, columnspan=2)

        self.window.protocol("WM_DELETE_WINDOW", self.onDestroy)
        self.window.mainloop()

    def onAddFace(self):
        student_code = self.entry_student_code.get()
        if checkStudentIfExits(student_code):
            add_face_data.run(student_code)
            self.entry_student_code.delete(0, 'end')
            mb.showinfo("Update face", "Updated face data successful")
        else:
            mb.showerror("Student's code not found", "Please enter a valid student's code")

    def onTrainModel(self):
        # Extract embedding
        extract_embeddings.run()

        # Train model
        train_model.run()
        mb.showinfo("Train model", "Model trained successful")

    def onDestroy(self):
        self.window.destroy()
        run()


def run():
    root = Tk()
    root.geometry('400x200+700+300')
    Authentication(root)

    root.mainloop()


run()
