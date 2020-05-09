import yagmail
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--receiver", required=True, help="Receiver email address")
ap.add_argument("-b", "--body", required=True, help="Email body")
ap.add_argument("-f", "--file", help="Attendance file")
args = vars(ap.parse_args())

receiver = args["receiver"]
body = args["body"]
filename = args["file"]
# filename = "Attendance" + os.sep + "Attendance_2019-08-29_13-09-07.csv"

# Mail information
yag = yagmail.SMTP("tieuluankhoahocmat3515@gmail.com", "ThisIsPassword")

# sent the mail
if filename is not None:
    yag.send(
        to=receiver,
        subject="Attendance Report",  # email subject
        contents=body,  # email body
        attachments=filename,  # file attached
    )
else:
    yag.send(
        to=receiver,
        subject="Attendance Report",  # email subject
        contents=body,  # email body
    )