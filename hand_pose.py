import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles


# Initialize MediaPipe hands model
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    # Initialize OpenCV video capture object
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB color space
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with MediaPipe hands model
        results = hands.process(frame_rgb)
        frameWithLandmarks = frame_rgb.copy()
        # Draw hand landmarks on frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw joints
                mp_drawing.draw_landmarks(frameWithLandmarks,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )


        # Display output frame
        cv2.imshow('Hand Pose Estimation', cv2.cvtColor(frameWithLandmarks, cv2.COLOR_BGR2RGB))

        # Exit program when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
