import cv2
import numpy as np
import onnxruntime
from playsound import playsound

# Load the ONNX model
model = onnxruntime.InferenceSession("last.onnx")

# Define a function to play a sound when the model detects an object
def play_sound_on_detection(output):
    # If the model detects an object, play a sound
    if len(output) > 0:
        playsound('sound.mp3')

# Define a function to pre-process the frame for the ONNX model
def preprocess_frame(frame):
    # Resize the frame to the input size of the ONNX model
    frame = cv2.resize(frame, (640, 640))
    frame = np.array([np.transpose(frame[:, :, 0]), np.transpose(frame[:, :, 1]), np.transpose(frame[:, :, 2])])
    # Convert the frame to a numpy array, normalize it, and add a batch dimension
    frame = (frame.astype('float32') / 255)[None, :, :, :]
    return frame

# Open the camera using cv2
camera = cv2.VideoCapture(0)
i = 0
# Continuously capture and process frames from the camera
while i == 0:
    i =1
    # Capture a frame from the camera
    _, frame = camera.read()
    
    # Pre-process the frame for the ONNX model
    input_data = preprocess_frame(frame)
    
    # Run the ONNX model to make a prediction
    output = model.run(None, {'images': input_data})[0]
    for row in output:
        print(' '.join(['{:.2f}+{:.2f}j'.format(val.real, val.imag) for val in row]))
    # Play a sound if the model detects an object
    # play_sound_on_detection(output)
    # Show the frame
    cv2.imshow('frame', frame)
    
    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy the window
camera.release()
cv2.destroyAllWindows()
