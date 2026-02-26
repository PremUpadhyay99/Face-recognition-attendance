import face_recognition
import os
import pickle
import cv2

dataset_dir = "dataset"
known_encodings = []
known_names = []

print("[INFO] Encoding faces...")

for person_name in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person_name)

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        image = cv2.imread(img_path)
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")

        if len(boxes) == 0:
            print(f"[WARNING] No face found in {img_path}")
            continue

        encodings = face_recognition.face_encodings(rgb, boxes)

        for enc in encodings:
            known_encodings.append(enc)
            known_names.append(person_name)
            print(f"[OK] Encoded {person_name} from {img_name}")

data = {
    "encodings": known_encodings,
    "names": known_names
}

with open("encodings.pickle", "wb") as f:
    pickle.dump(data, f)

print(f"[DONE] Total faces encoded: {len(known_encodings)}")
