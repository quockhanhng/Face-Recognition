import os


def mainMenu():
    os.system('cls')
    print()
    print(10 * "*", "WELCOME MENU", 10 * "*")
    print("[1] Add or update student's face data")
    print("[2] Train Images")
    print("[3] Recognize & Attendance")
    print("[4] Auto Mail")
    print("[0] Quit")

    while True:
        try:
            choice = int(input("Enter Choice: "))

            if choice == 1:
                student_id = str(input("Enter Student ID: "))
                AddFaceData(student_id1)
                break
            elif choice == 2:
                TrainImages()
                break
            elif choice == 3:
                class_id = str(input("Enter Class ID: "))
                RecognizeFaces(class_id)
                break
            elif choice == 4:
                SendMail()
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
    command = "py add_face_data.py --detector face_detection_model --student_id " + student_id
    os.system(command)
    mainMenu()


def TrainImages():
    # extract embedding
    command = "python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7"
    os.system(command)

    # train model
    command = "python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle"
    os.system(command)
    mainMenu()


def RecognizeFaces(class_id):
    command = "python recognize.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle --class_id " + class_id
    os.system(command)
    mainMenu()


def SendMail():
    # Send mail
    mainMenu()


mainMenu()
