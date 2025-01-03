import mediapipe as mp
import cv2
import numpy as np
import os
import shutil
import random

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

def split_dataset(data_dir, output_dir, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    """
    train: 80%, valid: 10%, test: 10%
    """
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "valid"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue

        images = [img for img in os.listdir(label_dir) if img.endswith(('.jpg', '.jpeg', '.npy', '.png'))]
        if len(images) == 0:
            continue

        random.shuffle(images)

        total_count = len(images)
        train_count = int(total_count * train_ratio)
        valid_count = int(total_count * valid_ratio)
        test_count = total_count - train_count - valid_count

        train_images = images[:train_count]
        valid_images = images[train_count:train_count+valid_count]
        test_images = images[train_count+valid_count:]

        if train_images:
            dst_dir = os.path.join(output_dir, "train", label)
            os.makedirs(dst_dir, exist_ok=True)
            for img_name in train_images:
                shutil.copy(os.path.join(label_dir, img_name), dst_dir)

        if valid_images:
            dst_dir = os.path.join(output_dir, "valid", label)
            os.makedirs(dst_dir, exist_ok=True)
            for img_name in valid_images:
                shutil.copy(os.path.join(label_dir, img_name), dst_dir)

        if test_images:
            dst_dir = os.path.join(output_dir, "test", label)
            os.makedirs(dst_dir, exist_ok=True)
            for img_name in test_images:
                shutil.copy(os.path.join(label_dir, img_name), dst_dir)


def extract_keypoints(image):

    h, w, _ = image.shape
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            return keypoints
    return np.zeros(21 * 3)

def augment_keypoints(keypoints):
    
    if not keypoints.any():
        return keypoints

    # random scaling
    scale = np.random.uniform(0.9, 1.1)
    keypoints *= scale

    # random rotation
    shift = np.random.uniform(-0.05, 0.05, size=keypoints.shape)
    keypoints += shift

    # random noise
    noise = np.random.normal(0, 0.01, size=keypoints.shape)
    keypoints += noise

    return keypoints

if __name__ == "__main__":
    dataset_dir = "./data/ASL/asl_alphabet_train"
    output_dir = "./data/processed_dataset_L"
    split_dir = "./data/processed_dataset_split_L"
    os.makedirs(output_dir, exist_ok=True)

    for label in os.listdir(dataset_dir):
        if label.startswith("."):  
            continue
        label_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_dir):  
            continue

        output_label_dir = os.path.join(output_dir, label)
        os.makedirs(output_label_dir, exist_ok=True)

        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            if not img_name.endswith(('.jpg', '.jpeg', '.png')):  
                continue

            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Unable to read {img_path}")
                continue

            keypoints = extract_keypoints(image)

            keypoints = augment_keypoints(keypoints)

            output_path = os.path.join(output_label_dir, img_name.rsplit('.', 1)[0] + '.npy')
            np.save(output_path, keypoints)

    
    split_dataset(output_dir, split_dir)
    print("Data preprocessing and splitting completed.")
