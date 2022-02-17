import cv2
import os
import matplotlib.pyplot as plt

image_face = r"Data/boy_face.jpg"
frontal_face = r"FaceDetection/haarcascade_frontalface_alt.xml"
os.chdir('../')
current_dir = os.getcwd()
image_path = os.path.join(current_dir, image_face)

image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# pre-trained viola-james algorithm include positive and negative
cascade_classifier = cv2.CascadeClassifier(os.path.join(current_dir, frontal_face))
detected_faces = cascade_classifier.detectMultiScale(gray_image, scaleFactor=1.07, minNeighbors=10, minSize=(30, 30))
print(detected_faces)

for x, y, width, height in detected_faces:
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 10)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
