import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Configuration
SAMPLE_RATE = 22050  # Sampling rate (matches training)
N_MFCC = 13  # Number of MFCCs to extract
MAX_LEN = 200  # Maximum number of timesteps

# Path to the saved model
MODEL_PATH = "rnn_gunshot_model.h5"

# Ensure the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at path: {MODEL_PATH}")

# Load the pre-trained model
print("Loading the trained model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")


# Function to load audio from a file
def load_audio(file_path, sample_rate):
    print(f"Loading audio from: "{D:\Project\ESC-50-master\ESC-50-master\audio\1-1791-A-26.wav}")
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
        print(f"Audio loaded successfully with shape: {audio.shape}")
        return audio
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None


# Function to extract MFCC features
def extract_features(audio, sample_rate, n_mfcc=13, max_len=200):
    try:
        # Trim silent parts
        audio, _ = librosa.effects.trim(audio, top_db=20)

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        print(f"Extracted MFCC shape: {mfcc.shape}")

        # Pad or truncate to fixed length
        if mfcc.shape[1] > max_len:
            mfcc = mfcc[:, :max_len]
            print(f"MFCC truncated to shape: {mfcc.shape}")
        else:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
            print(f"MFCC padded to shape: {mfcc.shape}")

        # Transpose and add batch dimension
        mfcc = mfcc.T  # Shape: (timesteps, n_mfcc)
        mfcc = np.expand_dims(mfcc, axis=0)  # Shape: (1, timesteps, n_mfcc)
        print(f"Final feature shape: {mfcc.shape}")

        return mfcc
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


# Function to make prediction
def predict_gunshot(features, model):
    try:
        prediction = model.predict(features)
        predicted_label = np.argmax(prediction, axis=1)
        confidence = np.max(prediction, axis=1)

        if predicted_label[0] == 1:
            return f"GUNSHOT detected with {confidence[0] * 100:.2f}% confidence."
        else:
            return f"NO gunshot detected with {confidence[0] * 100:.2f}% confidence."
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Prediction failed."


# Main function for file-based detection
def main():
    # Get the file path from the user
    file_path = input("Enter the path to the audio file: ").strip()

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Load audio from the file
    audio = load_audio(file_path, SAMPLE_RATE)

    if audio is None:
        print("Failed to load audio.")
        return

    # Extract features
    features = extract_features(audio, SAMPLE_RATE, N_MFCC, MAX_LEN)

    if features is None:
        print("Failed to extract features from the audio.")
        return

    # Make prediction
    result = predict_gunshot(features, model)

    # Output the result
    print(result)


if __name__ == "__main__":
    main()
