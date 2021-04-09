import cv2

haarcascade_file = '../haarcascades/haarcascade_frontalface_default.xml'
haarcascade = cv2.CascadeClassifier(haarcascade_file)
source = cv2.imread('../media/oscars.jpg')

# convertir fuente a escala de grises
gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

# detectar modelo
model = haarcascade.detectMultiScale(gray, 1.1, 4)

# Marcar el elemento en la imagen
for (x, y, w, h) in model:
    cv2.rectangle(source, (x, y), (x+w, y+h), (0, 255, 0), 2)

#Mostrar el resultado
cv2.imshow('img', source)
cv2.waitKey()