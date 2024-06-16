import os
import cv2
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt
from pose_detection import mediapipe_detection, draw_styled_landmarks, extract_keypoints

# Define the data path
DATA_PATH = 'Store'

# Define the actions
actions = np.array(['how are you', 'who are you', 'whats your name'])

# Number of sequences and sequence length
no_sequences = 5
sequence_length = 30  # Increased sequence length

# Create directories for new actions if they do not exist
for action in actions:
    action_folder = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_folder):
        os.makedirs(action_folder)
    dirmax = np.max(np.array(os.listdir(action_folder)).astype(int)) if len(os.listdir(action_folder)) > 0 else 0
    for sequence in range(1, no_sequences + 1):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(dirmax + sequence)))
        except:
            pass

# Collect keypoint values for training and testing
cap = cv2.VideoCapture(0)
with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    for action in actions:
        print(f"Press 'l' to start collecting frames for action: {action}")
        while True:
            ret, frame = cap.read()
            cv2.putText(frame, f'Press "l" to start collecting {action}', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('l'):
                break
        
        for sequence in range(1, no_sequences + 1):
            for frame_num in range(sequence_length):

                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)
                
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(500)
                else: 
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()
