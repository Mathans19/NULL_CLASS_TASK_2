import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
import pickle
import warnings
from PIL import Image, ImageTk
from collections import Counter
from tensorflow.keras.models import load_model
from pose_detection import mediapipe_detection, draw_styled_landmarks, extract_keypoints
import datetime

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# Load the hand gesture recognition model
model_dict = pickle.load(open('./model.p', 'rb'))
hand_model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Load the action recognition model
action_model = load_model('sign_detection.h5')

# Define labels dictionary for hand gestures including 'unknown' label
labels_dict = {0: 'A', 1: 'S', 2: 'L', -1: 'Unknown'}

# Define the maximum number of features expected for hand gestures
max_features = 42

# Define the actions for action recognition
actions = np.array(['how are you', 'who are you', 'whats your name'])

# Initialize detection variables for action recognition
sequence = []
all_predicted_actions = []

def process_image(image_path, label, image_label):
    # Check if the current time is between 6 PM and 10 PM
    now = datetime.datetime.now()
    if now.hour < 18 or now.hour >= 22:
        label.config(text="Image processing is only available between 6 PM and 10 PM.")
        return

    image = cv2.imread(image_path)
    predictions = []
    seen_predictions = set()

    def show_image():
        frame = image.copy()
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            data_aux = []
            for hand_landmarks in results.multi_hand_landmarks:
                x_, y_ = [], []
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            while len(data_aux) < max_features:
                data_aux.append(0)

            data_aux = np.asarray(data_aux[:max_features])

            prediction = hand_model.predict([data_aux])
            predicted_character = labels_dict[int(prediction[0])]
            cv2.putText(frame, predicted_character, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Append prediction to list if not already seen
            if predicted_character not in seen_predictions:
                predictions.append(predicted_character)
                seen_predictions.add(predicted_character)

        # Display the frame in the Tkinter window
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        image_label.imgtk = imgtk
        image_label.configure(image=imgtk)

        # Update the predictions label
        label.config(text=" ".join(predictions))

    show_image()

def browse_file(label, image_label):
    filepath = filedialog.askopenfilename()
    process_image(filepath, label, image_label)

def video_loop(video_path):
    global sequence, all_predicted_actions
    # Check if the current time is between 6 PM and 10 PM
    now = datetime.datetime.now()
    if now.hour < 18 or now.hour >= 22:
        sentence_label.config(text="Video processing is only available between 6 PM and 10 PM.")
        return

    # Initialize the video capture
    cap = cv2.VideoCapture(video_path)

    predictions = []

    # Set up Mediapipe model
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Read frames from the video capture
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = action_model.predict(np.expand_dims(sequence, axis=0))[0]
                predicted_action = actions[np.argmax(res)]
                all_predicted_actions.append(predicted_action)

                # Display actual and custom predictions separately
                predictions.append(predicted_action)

            # Convert image for Tkinter
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(image)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.config(image=imgtk)
            root.update_idletasks()
            root.update()

        # If video is completed, update the sentence label with the most predicted action
        most_common_action = Counter(predictions).most_common(1)[0][0]
        if most_common_action == 'who are you':
            custom_output = 'Custom: whats your name'
        elif most_common_action == 'whats your name':
            custom_output = 'Custom: who are you'
        else:
            custom_output = 'Custom: None'
        sentence_label.config(text=f"Actual: {most_common_action}\n{custom_output}")

def select_video():
    # Ask user to select a video file
    video_path = filedialog.askopenfilename()
    if video_path:
        # Call video_loop function with the selected video file
        video_loop(video_path)

# Create GUI window
root = tk.Tk()
root.title("Gesture and Action Recognition")

# Create label to display predictions for images
prediction_label = tk.Label(root, text="", font=("Helvetica", 16), wraplength=500)
prediction_label.pack(pady=20)

# Add image label to display the image
image_label = tk.Label(root)
image_label.pack()

# Add button to browse image file
browse_button = tk.Button(root, text="Browse Image File", command=lambda: browse_file(prediction_label, image_label))
browse_button.pack(pady=20)

# Create labels for video
video_label = tk.Label(root)
video_label.pack(padx=10, pady=10)

sentence_label = tk.Label(root, text="", font=("Helvetica", 16))
sentence_label.pack(padx=10, pady=10)

# Button to select video
select_button = tk.Button(root, text="Select Video File", command=select_video)
select_button.pack(padx=10, pady=10)

# Run the GUI
root.mainloop()
