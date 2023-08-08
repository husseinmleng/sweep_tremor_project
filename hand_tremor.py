import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import mediapipe as mp
import math
from flask import Flask, request, jsonify

def pixel_size(video_width, video_height, camera_resolution):
  #Calculates the pixel size for a video and camera.

  pixel_size_x = camera_resolution[0] / video_width
  pixel_size_y = camera_resolution[1] / video_height
  return math.sqrt(pixel_size_x * pixel_size_y)

def initialize_models():
    # Initialize MediaPipe hand detection and pose estimation models
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,min_tracking_confidence=0.5,model_complexity=1)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    return mp_hands, hands, mp_drawing, mp_drawing_styles

def load_video(video_path):
    # Load video file
    cap = cv2.VideoCapture(video_path)
    image_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    image_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    camera_resolution = (image_width ,image_height)
    return cap, camera_resolution, image_width, image_height

def detect_hand_landmarks(frame, hands):
    # Detect hand landmarks using MediaPipe
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(frame)
    return results_hands

def draw_hand_landmarks(frame, hand_landmarks, mp_hands, mp_drawing, mp_drawing_styles):
    # Draw hand landmarks on the frame
    frameWithLandmarks = frame.copy()
    mp_drawing.draw_landmarks(
        frameWithLandmarks,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )
    return frameWithLandmarks

def process_tremor_signal(tremor_signal, timestamps):
    # Process the tremor signal
    sampling_rate = 1000 / np.mean(np.diff(timestamps))
    time = np.array(timestamps) / 1000
    f, Pxx = signal.welch(tremor_signal, fs=sampling_rate, nperseg=1024)
    return time, f, Pxx

def plot_tremor_signal(time, tremor_signal_cm):
    # Plot the tremor signal and frequency spectrum
    fig, ax1 = plt.subplots(figsize=(8, 8))
    
    # Plot the tremor signal
    ax1.plot(time, tremor_signal_cm, label='Hand position')
    
    # Plot the range of tremor
    ax1.axhline(y=max(tremor_signal_cm), color='dimgrey', linestyle='dotted', label='Range of tremor')
    ax1.axhline(y=min(tremor_signal_cm), color='dimgrey', linestyle='dotted')
    
    # Plot the median amplitude
    ax1.axhline(y=np.median(tremor_signal_cm)/2, color='dimgrey', linestyle='dashed', label='Median amplitude')
    ax1.axhline(y=-np.median(tremor_signal_cm)/2, color='dimgrey', linestyle='dashed')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Tremor Amplitude (cm)')
    ax1.set_title('Waveform of a Tremor with a Measured Median Amplitude of %.2fcm' % np.median(tremor_signal_cm))
    
    leg = ax1.legend(ncol=3, bbox_to_anchor=(0.93, -0.15), fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(left=0.125, right=0.9, top=0.88, bottom=0.185)
    
    # Show the plot
    plt.show()


def main(video_path):
    # Initialize models
    mp_hands, hands, mp_drawing, mp_drawing_styles = initialize_models()
    return result

    # Load video
    cap, camera_resolution, image_width, image_height = load_video(video_path)

    # Initialize arrays for storing the tremor signal and timestamps
    tremor_signal = []
    timestamps = []

    # Initialize a variable to store the frame count
    frame_count = 0
    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Calculate the number of frames to process for the first 20 seconds
    frames_to_process = int(fps * 20)

    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        # If the frame count exceeds the number of frames to process, break the loop
        if frame_count > frames_to_process:
            break

        # Detect hand landmarks using MediaPipe
        results_hands = detect_hand_landmarks(frame, hands)

        frameWithLandmarks = frame.copy()
        if results_hands.multi_hand_landmarks:
            try:
                hand_landmarks = results_hands.multi_hand_landmarks[1]
            except:
                hand_landmarks = results_hands.multi_hand_landmarks[0]
            frameWithLandmarks = draw_hand_landmarks(frame, hand_landmarks, mp_hands, mp_drawing, mp_drawing_styles)

             # Compute the normalized hand landmark coordinates
            norm_hand_landmarks = np.array([(lmk.x, lmk.y, lmk.z) for lmk in hand_landmarks.landmark])
            norm_hand_landmarks[:, 2] = 0

            # Convert the normalized hand landmark coordinates to pixel coordinates
            img_h, img_w, _ = frame.shape
            px_hand_landmarks = np.zeros_like(norm_hand_landmarks)
            px_hand_landmarks[:, 0] = norm_hand_landmarks[:, 0] * img_w
            px_hand_landmarks[:, 1] = norm_hand_landmarks[:, 1] * img_h

            # Compute the centroid of the hand landmarks
            cx, cy = np.mean(px_hand_landmarks[:, :2], axis=0)

            # Draw the hand landmarks and centroid on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

            # Add the y-coordinate of the centroid to the tremor signal
            tremor_signal.append(cy)

            # Add the timestamp to the timestamps array
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Display the frame with hand landmarks
        cv2.imshow('Hand Tracking', frameWithLandmarks)

        # Increment the frame count
        frame_count += 1

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

    # Process the tremor signal
    time, f, Pxx = process_tremor_signal(tremor_signal, timestamps)

    # Convert tremor signal to cm
    pixelSize = pixel_size(image_width, image_height, camera_resolution)  
    tremor_signal_cm = [(x - min(tremor_signal) - ((max(tremor_signal) - min(tremor_signal)) / 2)) * pixelSize for x in tremor_signal]

    # Plot the tremor signal and frequency spectrum
    plot_tremor_signal(time, tremor_signal_cm)

from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400
    file = request.files['file']
    file.save(file.filename)
    if not os.path.exists(file.filename):
        return jsonify({'error': 'file not saved'}), 500
    result = main(file.filename)
    return jsonify({'result': result}), 200

if __name__ == "__main__":
    app.run()