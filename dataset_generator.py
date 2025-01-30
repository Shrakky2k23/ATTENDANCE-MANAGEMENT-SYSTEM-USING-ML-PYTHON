import numpy as np
import cv2
import sqlite3
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import datetime

Detector1 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
Detector2 = FaceMeshDetector(maxFaces=2)
cam = cv2.VideoCapture(0)


def insertorupdate(ID, NAME, DEPT, YEAR):
    conn = sqlite3.connect("database.db")
    cmd = "SELECT * FROM students WHERE ID=" + str(ID)
    cursor = conn.execute(cmd)
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
        break

    if isRecordExist == 1:
        conn.execute("UPDATE students SET NAME=? WHERE ID=?", (NAME, ID))
        conn.execute("UPDATE students SET DEPT=? WHERE ID=?", (DEPT, ID))
        conn.execute("UPDATE students SET YEAR=? WHERE ID=?", (YEAR, ID))
    else:
        conn.execute("INSERT INTO students(ID,NAME,DEPT,YEAR) VALUES (?,?,?,?)", (ID, NAME, DEPT, YEAR))

    conn.commit()
    conn.close()


ID = input('Enter User id: ')
NAME = input('Enter User name: ')
DEPT = input('Enter Department name: ')
YEAR = input('Enter Academic year: ')

insertorupdate(ID, NAME, DEPT, YEAR)

sampleNum = 0
while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = Detector1.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        sampleNum += 1
        cv2.imwrite(f"datasets/user.{ID}.{sampleNum}.jpg", gray[y:y + h, x:x + w])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.waitKey(100)

        now = datetime.datetime.now()
        attendance_time = now.strftime("%H:%M:%S")
        print(f"Attendance marked at {attendance_time}")

    cv2.imshow("FACE", img)
    if cv2.waitKey(1) & 0xFF == ord('q') or sampleNum > 20:
        break

while True:
    success, img = cam.read()
    if not success:
        print("Failed to grab frame")
        break

    now = datetime.datetime.now()
    time_str = now.strftime("%H:%M:%S")
    cv2.putText(img, f"Time: {time_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    img, faces = Detector2.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]

        x_min = min(point[0] for point in face)
        y_min = min(point[1] for point in face)
        x_max = max(point[0] for point in face)
        y_max = max(point[1] for point in face)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        w, _ = Detector2.findDistance(pointLeft, pointRight)
        W = 6.3
        f = 303.735
        d = (f * W) / w
        print(f"Distance: {d:.2f} cm")

        if len(face) > 10:
            cvzone.putTextRect(img, f'DISTANCE: {d:.2f}cm', (face[10][0] - 100, face[10][1] - 50), scale=2,
                               colorR=(0, 0, 255))

    cv2.imshow("Shirsha", img)
    if cv2.waitKey(1) & 0xFF == ord('R'):
        break

cam.release()
cv2.destroyAllWindows()