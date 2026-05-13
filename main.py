import cv2
import os
import numpy as np
import pyttsx3
import speech_recognition as sr
import webbrowser
import threading
import time

# ---------------- VOICE SYSTEM ----------------
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

is_speaking = False

def _say(text):
    global is_speaking
    is_speaking = True
    print("VOICE:", text)
    engine.say(text)
    engine.runAndWait()
    is_speaking = False

def speak(text):
    global is_speaking
    if not is_speaking:
        threading.Thread(target=_say, args=(text,), daemon=True).start()

# ---------------- FACE MODEL ----------------
recognizer = cv2.face.LBPHFaceRecognizer_create()

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- LOAD DATASET ----------------
dataset_path = "dataset"

faces = []
labels = []
names = {}
label_id = 0

for person in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person)

    if os.path.isdir(person_path):

        names[label_id] = person

        for img_name in os.listdir(person_path):

            img_path = os.path.join(person_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:

                detected = face_detector.detectMultiScale(
                    img,
                    scaleFactor=1.1,
                    minNeighbors=6,
                    minSize=(80,80)
                )

                for (x, y, w, h) in detected:
                    face = img[y:y+h, x:x+w]
                    face = cv2.equalizeHist(face)
                    face = cv2.resize(face, (200,200))
                    faces.append(face)
                    labels.append(label_id)

        label_id += 1

if len(faces) > 0:
    recognizer.train(faces, np.array(labels))
else:
    print("No dataset found")
    exit()

# ---------------- DISTANCE ----------------
def estimate_distance(w):
    if w == 0:
        return 0
    return round(500 / w, 2)

# ---------------- DESTINATION ----------------
def get_destination():

    r = sr.Recognizer()

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        speak("You said " + text)
        return text

    except:
        speak("Could not understand")
        return None

# ---------------- GOOGLE MAP ----------------
def open_map(place):

    url = (
        "https://www.google.com/maps/dir/?api=1&destination="
        + place.replace(" ", "+")
        + "&travelmode=walking"
    )

    webbrowser.open(url)
    speak("Opening directions to " + place)

# ---------------- MAIN ----------------
def main():

    cap = cv2.VideoCapture(0)

    speak("Assistive system started")
    time.sleep(3)

    speak("Say your destination")
    time.sleep(3)

    place = get_destination()

    if place:
        open_map(place)

    last_spoken_name = ""
    last_spoken_time = 0
    last_obstacle_time = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---------------- FACE RECOGNITION ----------------
        detected_faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(80,80)
        )

        for (x, y, w, h) in detected_faces:

            face = gray[y:y+h, x:x+w]
            face = cv2.equalizeHist(face)
            face = cv2.resize(face, (200,200))

            label, confidence = recognizer.predict(face)

            if confidence < 62:
                name = names[label]
            else:
                name = "Unknown Person"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            cv2.putText(
                frame,
                name,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

            current_time = time.time()

            if name != last_spoken_name or current_time - last_spoken_time > 6:

                if name == "Unknown Person":
                    speak("Unknown person detected")
                else:
                    speak(name + " is in front of you")

                last_spoken_name = name
                last_spoken_time = current_time

        # ---------------- MULTI OBSTACLE DETECTION ----------------
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)

        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        obstacle_list = []

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area > 3000:

                x, y, w, h = cv2.boundingRect(cnt)

                if h < 40 or w < 40:
                    continue

                distance = estimate_distance(w)

                center_x = x + w // 2

                if center_x < frame.shape[1] // 3:
                    side = "left"
                elif center_x < 2 * frame.shape[1] // 3:
                    side = "center"
                else:
                    side = "right"

                obstacle_list.append((distance, side, x, y, w, h))

        obstacle_list.sort(key=lambda item: item[0])

        for item in obstacle_list[:3]:

            distance, side, x, y, w, h = item

            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

            cv2.putText(
                frame,
                f"{side} {distance}m",
                (x,y-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,255),
                2
            )

        if len(obstacle_list) > 0:

            nearest = obstacle_list[0]
            distance = nearest[0]
            side = nearest[1]

            if distance < 2.5:

                if time.time() - last_obstacle_time > 4:
                    speak("Obstacle on " + side)
                    last_obstacle_time = time.time()

        cv2.imshow("Assistive Camera", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()