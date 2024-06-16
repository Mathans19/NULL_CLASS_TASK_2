import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import TensorBoard


# Define the actions
actions = np.array(['how are you', 'who are you', 'whats your name'])
label_map = {label: num for num, label in enumerate(actions)}

# Prepare data sequences and labels
DATA_PATH = 'Store'  # Replace with the actual data path
sequence_length = 30  # Increased sequence length
sequences, labels = [], []

for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Check if the test set has enough samples
if len(X_test) < 5:
    print("Test set has fewer samples. Adjusting...")
    num_samples = min(5, len(X_test))
    X_test_adjusted = X_test[:num_samples]
    y_test_adjusted = y_test[:num_samples]
else:
    X_test_adjusted = X_test
    y_test_adjusted = y_test

# Build the LSTM neural network
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(Input(shape=(sequence_length, X.shape[2])))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

# Print the model summary
model.summary()

# Make predictions
res = model.predict(X_test_adjusted)
print(f"Predicted: {actions[np.argmax(res[0])]}")
print(f"Actual: {actions[np.argmax(y_test_adjusted[0])]}")

# Save model weights
model.save('sign_detection.h5')
