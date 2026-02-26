import cv2
import mediapipe as mp
import numpy as np
import os
import csv
from datetime import datetime

# -------------------- MEDIAPIPE SETUP --------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# -------------------- CAMERA --------------------
cap = cv2.VideoCapture(0)

# -------------------- LANDMARKS --------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14]

def dist(a, b):
    return np.linalg.norm(a - b)

def EAR(points, eye):
    return (dist(points[eye[1]], points[eye[5]]) +
            dist(points[eye[2]], points[eye[4]])) / \
           (2.0 * dist(points[eye[0]], points[eye[3]]))

# -------------------- FACE RECOGNITION --------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
faces, labels = [], []
label_map = {}
label_id = 0

dataset_path = "dataset"

for person in os.listdir(dataset_path):
    label_map[label_id] = person
    person_path = os.path.join(dataset_path, person)
    for img_name in os.listdir(person_path):
        img = cv2.imread(os.path.join(person_path, img_name), 0)
        faces.append(img)
        labels.append(label_id)
    label_id += 1

recognizer.train(faces, np.array(labels))

# -------------------- ATTENDANCE --------------------
attendance_file = "attendance.csv"
marked = set()

def mark_attendance(name):
    if name not in marked:
        now = datetime.now()
        with open(attendance_file, 'a', newline='') as f:
            csv.writer(f).writerow([name, now.date(), now.strftime("%H:%M:%S")])
        marked.add(name)

# -------------------- LIVENESS VARIABLES --------------------
blink_counter = 0
closed_frames = 0

# -------------------- MAIN LOOP --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        points = np.array([(int(p.x * w), int(p.y * h)) for p in lm])

        # EAR, mouth, tilt
        ear = (EAR(points, LEFT_EYE) + EAR(points, RIGHT_EYE)) / 2
        mouth = dist(points[MOUTH[0]], points[MOUTH[1]])
        tilt = abs(points[1][0] - points[10][0])

        # Blink logic
        if ear < 0.20:
            closed_frames += 1
        else:
            if closed_frames > 2:
                blink_counter += 1
            closed_frames = 0

        # Face ROI (safe crop)
        x1, y1 = max(points[234][0], 0), max(points[234][1], 0)
        x2, y2 = min(points[454][0], w), min(points[454][1], h)
        face_img = gray[y1:y2, x1:x2]

        is_real = False
        name = "Unknown"

        if face_img.size != 0:
            # Texture check (anti-phone)
            lap_var = cv2.Laplacian(face_img, cv2.CV_64F).var()

            # FINAL LIVENESS RULE
            if blink_counter >= 2 and mouth > 12 and lap_var > 30:
                is_real = True

            try:
                label, confidence = recognizer.predict(face_img)
                if confidence < 80:
                    name = label_map[label]
            except:
                pass

        if is_real and name != "Unknown":
            cv2.putText(frame, f"REAL - {name}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mark_attendance(name)
        else:
            cv2.putText(frame, "FAKE / PHOTO DETECTED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    else:
        cv2.putText(frame, "NO FACE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Liveness Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
