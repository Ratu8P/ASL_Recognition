import cv2
import torch
import numpy as np
from model import KeypointCNN as CNN

import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
model_dir = "./keypoint_model_L_cnn_50.pth"

# 加载类别映射
def load_class_mapping(mapping_file="class_mapping.txt"):
    class_mapping = {}
    with open(mapping_file, "r") as f:
        for line in f:
            idx, cls_name = line.strip().split(": ")
            class_mapping[int(idx)] = cls_name
    return class_mapping

model = CNN(num_classes=26)
model.load_state_dict(torch.load(model_dir, map_location="cpu"))
model.eval()

# load class mapping
class_mapping = load_class_mapping("class_mapping.txt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # display the class label
            landmarks = hand_landmarks.landmark
            h, w, _ = frame.shape
            min_y = min([lm.y for lm in landmarks])
            top_y = int(min_y * h)
            wrist_x = int(landmarks[0].x * w)

            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
            keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                output = model(keypoints_tensor)
                predicted_class = torch.argmax(output, dim=1).item()

            predicted_label = class_mapping.get(predicted_class, "Unknown")

    
            cv2.putText(frame, f"Class: {predicted_label}", (wrist_x, top_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No hand detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
