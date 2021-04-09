import cv2

haarcascade = cv2.CascadeClassifier('../haarcascades/haarcascade_eye.xml')
videcapture = cv2.VideoCapture(0)

while True:
    _, img = videcapture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = haarcascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('img', img)

    k = cv2.waitKey(30)
    if k == 27:
        break

videcapture.release()
