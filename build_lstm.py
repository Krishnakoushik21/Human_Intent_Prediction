import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
model = tf.keras.models.load_model("intent_lstm_model.h5")

INTENTS = [
    "Neutral",
    "Reach Forward",
    "Reach Up",
    "Reach Side",
    "Sit Down Start",
    "Stand Up Start",
    "Walk Start",
    "Turn Left",
    "Turn Right",
    "Lean Forward",
    "Lean Backward",
    "Loss of Balance"
]

SEQUENCE_LENGTH = 30

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

sequence = []
recording = False
prediction_ready = False

predicted_intent = "Waiting..."
confidence = 0.0

print("""
CONTROLS:
R - Start recording (30 frames)
P - Predict intent
Q - Quit
""")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        pose_vector = []
        for lm in results.pose_landmarks.landmark:
            pose_vector.extend([lm.x, lm.y, lm.z])

        if recording and len(pose_vector) == 99:
            sequence.append(pose_vector)

            if len(sequence) == SEQUENCE_LENGTH:
                recording = False
                prediction_ready = True
                print("Recording complete. Press P to predict.")

    key = cv2.waitKey(1) & 0xFF

    # START RECORDING
    if key == ord('r') and not recording:
        sequence = []
        recording = True
        prediction_ready = False
        predicted_intent = "Recording..."
        print("Recording started...")

    # PREDICT ONLY WHEN USER ASKS
    if key == ord('p') and prediction_ready:
        input_data = np.expand_dims(sequence, axis=0)
        preds = model.predict(input_data, verbose=0)

        class_id = np.argmax(preds)
        confidence = preds[0][class_id]
        predicted_intent = INTENTS[class_id]

        print(f"Predicted: {predicted_intent} ({confidence*100:.1f}%)")
        prediction_ready = False

    # DISPLAY
    cv2.rectangle(frame, (0, 0), (480, 90), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Intent: {predicted_intent}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence*100:.1f}%",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.imshow("Real-Time Human Intent Prediction", frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()