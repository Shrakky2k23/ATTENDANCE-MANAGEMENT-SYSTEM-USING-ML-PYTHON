import cv2
import numpy as np
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import datetime
import sqlite3

# Load the trained face recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer/trainingdata.yml")

# Initialize face detectors
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_mesh_detector = FaceMeshDetector(maxFaces=2)

# Initialize the webcam
cam = cv2.VideoCapture(0)

# Function to get the details of the recognized faceQ
def get_details(id):
    conn = sqlite3.connect("database.db")
    cmd = "SELECT NAME, DEPT, YEAR FROM students WHERE ID=" + str(id)
    cursor = conn.execute(cmd)
    details = None
    for row in cursor:
        details = row
    conn.close()
    return details

attendance_info = []

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Distance measurement
    img, faces_mesh = face_mesh_detector.findFaceMesh(img, draw=False)
    if faces_mesh:
        face_mesh = faces_mesh[0]
        point_left = face_mesh[145]
        point_right = face_mesh[374]

        # Distance calculation
        w, _ = face_mesh_detector.findDistance(point_left, point_right)
        W = 6.3  # Average pupillary distance
        f = 303.735  # Focal length calculated earlier
        d = (f * W) / w
        cvzone.putTextRect(img, f'DISTANCE: {d:.2f} cm', (10, 30), scale=1, colorR=(0, 0, 255), thickness=2)

    # Display current time
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    cvzone.putTextRect(img, f'TIME: {current_time}', (10, 60), scale=1, colorR=(0, 0, 255), thickness=2)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle
        face_id, _ = recognizer.predict(gray[y:y + h, x:x + w])
        details = get_details(face_id)

        if details:
            name, dept, year = details
            attendance_time = datetime.datetime.now().strftime("%H:%M:%S")
            cvzone.putTextRect(img, f"Name: {name}", (x, y + h + 30), scale=1, colorR=(0, 0, 255), colorB=(0, 255, 255), thickness=2)
            cvzone.putTextRect(img, f"ID: {face_id}", (x, y + h + 60), scale=1, colorR=(0, 0, 255), colorB=(0, 255, 255), thickness=2)
            cvzone.putTextRect(img, f"Dept: {dept}, Year: {year}", (x, y + h + 90), scale=1, colorR=(0, 0, 255), colorB=(0, 255, 255), thickness=2)
            cvzone.putTextRect(img, f"Time: {attendance_time}", (x, y + h + 120), scale=1, colorR=(0, 0, 255), colorB=(0, 255, 255), thickness=2)

            # Store attendance info
            attendance_info.append((name, attendance_time, d))
        else:
            cvzone.putTextRect(img, "Unknown", (x, y - 5), scale=1, colorR=(0, 0, 255), colorB=(0, 255, 255), thickness=2)

    cv2.imshow("Detector", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

# Print attendance info
print("\nAttendance Marked:")
for info in attendance_info:
    print(f"Name: {info[0]}, Time: {info[1]}, Distance: {info[2]:.2f} cm")
