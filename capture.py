import cv2
import os

name = input("Enter Name: ")

path = "faces/" + name
os.makedirs(path, exist_ok=True)

cam = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("Capture Face", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        count += 1
        cv2.imwrite(f"{path}/{count}.jpg", frame)
        print("Saved", count)

    elif key == 27:
        break

cam.release()
cv2.destroyAllWindows()