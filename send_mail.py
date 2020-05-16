import yagmail


def runWithAttachment(receiver, body, filename):
    # Mail information
    yag = yagmail.SMTP("tieuluankhoahocmat3515@gmail.com", "ThisIsPassword")

    # sent the mail
    yag.send(
        to=receiver,
        subject="Attendance Report",  # email subject
        contents=body,  # email body
        attachments=filename,  # file attached
    )


def runWithoutAttachment(receiver, body):
    # Mail information
    yag = yagmail.SMTP("tieuluankhoahocmat3515@gmail.com", "ThisIsPassword")

    yag.send(
        to=receiver,
        subject="Attendance Report",  # email subject
        contents=body,  # email body
    )
