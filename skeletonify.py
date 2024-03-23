import cv2
import mediapipe as mp
import os


base_folder = r"C:\Users\golba\OneDrive\Desktop\ECE176\datasets\archive\asl-alphabet-test"


save_base_folder = r"C:\Users\golba\OneDrive\Desktop\ECE176\final\skeletonimagestest"
if not os.path.exists(save_base_folder):
    os.makedirs(save_base_folder)

mp_hands = mp.solutions.hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Iterate over each subfolder in the base folder
for folder in os.listdir(base_folder):
    current_folder = os.path.join(base_folder, folder)
    save_folder = os.path.join(save_base_folder, folder)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Process each image in the current subfolder
    for image_name in os.listdir(current_folder):
        image_path = os.path.join(current_folder, image_name)

        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        # Process the image
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb_img)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Save the processed image
        save_path = os.path.join(save_folder, image_name)
        cv2.imwrite(save_path, gray_img)
        print(f'Saved {save_path}')

cv2.destroyAllWindows()
