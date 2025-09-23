import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Dataset paths
gunshot_paths = [
    r"D:\Project\edge-collected-gunshot-audio\edge-collected-gunshot-audio\38s&ws_dot38_caliber",
    r"D:\Project\edge-collected-gunshot-audio\edge-collected-gunshot-audio\glock_17_9mm_caliber",
    r"D:\Project\edge-collected-gunshot-audio\edge-collected-gunshot-audio\remington_870_12_gauge",
    r"D:\Project\edge-collected-gunshot-audio\edge-collected-gunshot-audio\ruger_ar_556_dot223_caliber"
]
non_gunshot_path = r"D:\Project\ESC-50-master\ESC-50-master\audio"

# Validate paths
def get_valid_files(paths):
    valid_files = []
    for path in paths:
        if os.path.exists(path):
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.wav')]
            valid_files.extend(files)
        else:
            print(f"Warning: Path does not exist - {path}")
    return valid_files

gunshot_files = get_valid_files(gunshot_paths)
non_gunshot_files = get_valid_files([non_gunshot_path])

if not gunshot_files or not non_gunshot_files:
    print("Error: No valid audio files found in the specified paths.")
    exit(1)

# Feature extraction
def extract_features(audio_file, n_mfcc=13, max_len=200):
    try:
        y, sr = librosa.load(audio_file, sr=22050)
        y = librosa.effects.trim(y, top_db=20)[0]  # Trim silent parts
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # Shape: (n_mfcc, time)
        # Pad or truncate to fixed length
        if mfcc.shape[1] > max_len:
            mfcc = mfcc[:, :max_len]
        else:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return mfcc.T  # Shape: (time, n_mfcc)
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

# Extract features
print("Extracting gunshot features...")
X_gunshot = [extract_features(f) for f in gunshot_files if f]
X_gunshot = [x for x in X_gunshot if x is not None]

print("Extracting non-gunshot features...")
X_non_gunshot = [extract_features(f) for f in non_gunshot_files if f]
X_non_gunshot = [x for x in X_non_gunshot if x is not None]

y_gunshot = [1] * len(X_gunshot)
y_non_gunshot = [0] * len(X_non_gunshot)

# Combine data
X = np.array(X_gunshot + X_non_gunshot)
y = np.array(y_gunshot + y_non_gunshot)

if len(X) == 0:
    print("No valid audio features extracted.")
    exit(1)

# One-hot encode labels
y = to_categorical(y, num_classes=2)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RNN model
model = Sequential([
    Masking(mask_value=0., input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64, return_sequences=False, activation='tanh'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Train the model
print("Training model...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Evaluate the model
print("Evaluating model...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Classification report
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Save the model
model.save('rnn_gunshot_model.h5')
