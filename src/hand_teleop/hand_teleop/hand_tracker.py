import cv2
import mediapipe as mp

# 1. Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# 2. Define the Finger Tip and Knuckle Pairs
# Order: Index, Middle, Ring, Pinky
TIPS = [8, 12, 16, 20]
KNUCKLES = [6, 10, 14, 18]

def get_finger_states(hand_landmarks):
    """Returns a list of 1s (open) and 0s (closed) for the 4 fingers."""
    states = []
    for i in range(4):
        tip_y = hand_landmarks.landmark[TIPS[i]].y
        knuckle_y = hand_landmarks.landmark[KNUCKLES[i]].y
        
        # If the tip is higher on the screen (lower Y value) than the knuckle, it is open
        if tip_y < knuckle_y:
            states.append(1) # Open
        else:
            states.append(0) # Closed
    return states

def classify_gesture(finger_states):
    """Maps the binary array to a human-readable command."""
    if finger_states == [0, 0, 0, 0]:
        return "COMMAND: STOP (Fist)"
    elif finger_states == [1, 1, 1, 1]:
        return "COMMAND: RESET ARM (Open Hand)"
    elif finger_states == [1, 0, 0, 0]:
        return "COMMAND: MOVE FORWARD (Pointing)"
    elif finger_states == [1, 1, 0, 0]:
        return "COMMAND: GRIPPER TOGGLE (Peace Sign)"
    else:
        return "UNKNOWN GESTURE"

# 3. Start the Video Loop
cap = cv2.VideoCapture(0)
print("Starting Camera... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract States and Classify
            f_states = get_finger_states(hand_landmarks)
            gesture = classify_gesture(f_states)
            
            # Display the data
            cv2.putText(frame, f"State Array: {f_states}", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, gesture, (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow('Gesture Command Interface', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()