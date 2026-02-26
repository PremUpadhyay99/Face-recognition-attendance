import cv2
import os

name = input("Enter person name: ").strip()
dataset_path = "dataset"
person_path = os.path.join(dataset_path, name)

os.makedirs(person_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Press 'c' to capture face, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Capture Face", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        img_path = os.path.join(person_path, f"{count}.jpg")
        cv2.imwrite(img_path, gray)
        print(f"Saved {img_path}")
        count += 1

    elif key == ord('q') or count >= 20:
        break

cap.release()
cv2.destroyAllWindows()
print("Face capture completed")
