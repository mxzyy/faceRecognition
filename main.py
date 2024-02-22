import cv2

def drawFace(frame, faces):
    for (x,y,w,h) in faces:
        face_recognition = frame[y:y+h, x:x+h]
        frame[y:y+h, x:x+h] = face_recognition
        cv2.rectangle(frame, (x, y), (x + w,y + h), (0, 0, 255), 2)
    return frame

url = 'http://192.168.137.148:8080/video'
cap = cv2.VideoCapture(url)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
detected_faces = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    detected_faces = faces
    frame = drawFace(frame, detected_faces)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
