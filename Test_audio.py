import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Path to the saved model
model_path = "rnn_gunshot_model.h5"

# Path to the input audio file
audio_file_path = r"D:\Project\edge-collected-gunshot-audio\edge-collected-gunshot-audio\ruger_ar_556_dot223_caliber\1d815f56-4309-4074-8bdb-cedaf6c0acc9_v2.wav"  # Update this to your actual file path if it's elsewhere

# Load the pre-trained model
model = load_model(model_path)

# Function to extract features from an audio file
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
        padded_mfcc = np.zeros((200, 13))  # Assuming 200 time steps
        padded_mfcc[:min(200, mfcc.shape[1]), :] = mfcc.T[:200]
        return np.expand_dims(padded_mfcc, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Extract features from the input audio file
features = extract_features(audio_file_path)

if features is not None:
    # Make prediction
    prediction = model.predict(features)
    predicted_label = np.argmax(prediction, axis=1)
    confidence = np.max(prediction, axis=1)

    # Output the result
    if predicted_label == 1:
        print(f"The audio is predicted to be a GUNSHOT with {confidence[0]*100:.2f}% confidence.")
    else:
        print(f"The audio is predicted to be a NON-GUNSHOT sound with {confidence[0]*100:.2f}% confidence.")
else:
    print("Could not extract features from the audio file.")
