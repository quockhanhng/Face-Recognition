import os


def mainMenu():
    os.system('cls')
    print()
    print(10 * "*", "WELCOME MENU", 10 * "*")
    print("[1] Add or update student")
    print("[2] Train Images")
    print("[3] Recognize & Attendance")
    print("[4] Auto Mail")
    print("[0] Quit")

    while True:
        try:
            choice = int(input("Enter Choice: "))

            if choice == 1:
                CaptureFaces()
                break
            elif choice == 2:
                TrainImages()
                break
            elif choice == 3:
                RecognizeFaces()
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

def CaptureFaces():
    os.system("py DataSetCreator.py")
    mainMenu()


def TrainImages():
    os.system("py Trainer.py")
    mainMenu()


def RecognizeFaces():
    os.system("py FaceDetection.py")
    mainMenu()


def SendMail():
    # Send mail
    mainMenu()


mainMenu()
