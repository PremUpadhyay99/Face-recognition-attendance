import cv2
import face_recognition
import mediapipe as mp
import numpy as np
import pickle
import csv
from datetime import datetime
import winsound
import pyttsx3
import os
import random
import time

# ================= PERFORMANCE =================
FRAME_SKIP = 4
RESIZE_WIDTH = 640

# ================= VOICE =================
engine = pyttsx3.init()
engine.setProperty("rate", 160)

# ================= LOAD ENCODINGS =================
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

# ================= MEDIAPIPE =================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

# ================= LIVENESS =================
EAR_THRESHOLD = 0.30
BLINK_FRAMES = 2
HEAD_MOVE_THRESHOLD = 0.03

# ================= RECOGNITION =================
FACE_DISTANCE_THRESHOLD = 0.65

# ================= STATE =================
blink_counter = 0
blink_detected = False
head_verified = False
head_start_x = None
head_positions = []

name_buffer = []
marked_names = set()
frame_count = 0
prev_time = time.time()

challenge_direction = random.choice(["LEFT", "RIGHT"])

# ================= ATTENDANCE FILE =================
CSV_FILE = "attendance.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Name", "Time", "Date"])

# ================= FUNCTIONS =================
def eye_aspect_ratio(pts, eye):
    A = np.linalg.norm(pts[eye[1]] - pts[eye[5]])
    B = np.linalg.norm(pts[eye[2]] - pts[eye[4]])
    C = np.linalg.norm(pts[eye[0]] - pts[eye[3]])
    return (A + B) / (2.0 * C)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not accessible")
    exit()

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(
        frame,
        (RESIZE_WIDTH, int(frame.shape[0] * RESIZE_WIDTH / frame.shape[1]))
    )
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_count += 1
    run_recognition = frame_count % FRAME_SKIP == 0

    boxes, encodings = [], []
    if run_recognition:
        small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
        boxes = face_recognition.face_locations(small, model="hog")
        encodings = face_recognition.face_encodings(small, boxes)
        boxes = [(t*2, r*2, b*2, l*2) for (t, r, b, l) in boxes]

    # ================= FACE MATCHING =================
    current_name = "Unknown"

    if encodings:
        enc = encodings[0]
        distances = face_recognition.face_distance(data["encodings"], enc)
        best = np.argmin(distances)

        if distances[best] < FACE_DISTANCE_THRESHOLD:
            name_buffer.append(data["names"][best])
        else:
            name_buffer.append("Unknown")

        if len(name_buffer) > 7:
            name_buffer.pop(0)

        if len(name_buffer) >= 5:
            most_common = max(set(name_buffer), key=name_buffer.count)
            if name_buffer.count(most_common) >= 3:
                current_name = most_common

    # ================= LIVENESS =================
    if boxes and run_recognition:
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            pts = np.array([[int(p.x*w), int(p.y*h)] for p in lm.landmark])

            ear = (eye_aspect_ratio(pts, LEFT_EYE) +
                   eye_aspect_ratio(pts, RIGHT_EYE)) / 2

            # ---- BLINK ----
            if ear < EAR_THRESHOLD:
                blink_counter += 1
            elif blink_counter >= BLINK_FRAMES:
                blink_detected = True
                blink_counter = 0

            # ---- HEAD MOVE ----
            if blink_detected and not head_verified:
                lx = lm.landmark[LEFT_CHEEK].x
                rx = lm.landmark[RIGHT_CHEEK].x
                cx = (lx + rx) / 2

                if head_start_x is None:
                    head_start_x = cx

                head_positions.append(cx)
                if len(head_positions) > 5:
                    head_positions.pop(0)

                diff = np.mean(head_positions) - head_start_x

                if challenge_direction == "LEFT" and diff < -HEAD_MOVE_THRESHOLD:
                    head_verified = True
                elif challenge_direction == "RIGHT" and diff > HEAD_MOVE_THRESHOLD:
                    head_verified = True

    # ================= ATTENDANCE =================
    if current_name != "Unknown" and blink_detected and head_verified:
        if current_name not in marked_names:
            marked_names.add(current_name)
            now = datetime.now()

            with open(CSV_FILE, "a", newline="") as f:
                csv.writer(f).writerow([
                    current_name,
                    now.strftime("%H:%M:%S"),
                    now.strftime("%d-%m-%Y")
                ])

            winsound.Beep(1200, 300)
            engine.say("Attendance marked successfully")
            engine.runAndWait()

            blink_detected = False
            head_verified = False
            head_start_x = None
            head_positions.clear()
            challenge_direction = random.choice(["LEFT", "RIGHT"])

    # ================= UI =================
    if boxes:
        t, r, b, l = boxes[0]
        color = (0, 255, 0) if current_name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (l, t), (r, b), color, 2)
        cv2.putText(frame, current_name, (l, t-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    status = (
        "Please Blink" if not blink_detected else
        f"Turn Head {challenge_direction}" if not head_verified else
        "Liveness Verified"
    )

    cv2.putText(frame, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0) if head_verified else (0, 0, 255), 2)

    curr_time = time.time()
    fps = int(1 / max(curr_time - prev_time, 0.001))
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
