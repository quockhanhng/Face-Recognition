import os
import add_face_data
import extract_embeddings
import train_model
import recognize_ui


def mainMenu():
    os.system('cls')
    print()
    print(10 * "*", "Student Attendance Management System", 10 * "*")
    print("[1] Add or update student's face data")
    print("[2] Train Images")
    print("[3] Recognize & Attendance")
    print("[0] Quit")

    while True:
        try:
            choice = int(input("Enter Choice: "))

            if choice == 1:
                student_id = str(input("Enter Student ID: "))
                AddFaceData(student_id)
                break
            elif choice == 2:
                TrainImages()
                break
            elif choice == 3:
                class_id = str(input("Enter Class ID: "))
                RecognizeFaces(class_id)
                # RecognizeUI()
                break
            elif choice == 0:
                print("Thank You")
                break
            else:
                print("Invalid Choice. Enter 1-4")
                mainMenu()
        except ValueError:
            print("Invalid Choice. Enter 1-4\n Try Again")
    exit()


# ---------------------------------------------------------

def AddFaceData(student_id):
    add_face_data.run(student_id)
    mainMenu()


def TrainImages():
    # Extract embedding
    extract_embeddings.run()

    # Train model
    train_model.run()
    mainMenu()


def RecognizeFaces(class_id):
    command = "python recognize.py --class_id " + class_id
    os.system(command)
    mainMenu()


def RecognizeUI():
    recognize_ui.run()
    mainMenu()


mainMenu()
