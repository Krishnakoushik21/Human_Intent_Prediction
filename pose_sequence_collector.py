import cv2
import mediapipe as mp
import numpy as np
import os

# ---------------- POSE DETECTOR ----------------
class PoseDetector:
    def __init__(self):
        self.mpPose = mp.solutions.pose # type: ignore
        self.pose = self.mpPose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mpDraw = mp.solutions.drawing_utils # type: ignore

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(
                img,
                self.results.pose_landmarks,
                self.mpPose.POSE_CONNECTIONS
            )
        return img

    def getPoseVector(self):
        pose_vector = []
        if self.results.pose_landmarks:
            for lm in self.results.pose_landmarks.landmark:
                pose_vector.extend([lm.x, lm.y, lm.z])
        return pose_vector  # 99 values


# ---------------- MAIN ----------------
def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    SEQUENCE_LENGTH = 30
    sequence = []
    recording = False

    os.makedirs("data", exist_ok=True)

    # --------- LOAD EXISTING DATA ----------
    X_path = "data/X.npy"
    y_path = "data/y.npy"

    if os.path.exists(X_path) and os.path.exists(y_path):
        X = list(np.load(X_path))
        y = list(np.load(y_path))
        print(f"Loaded existing data: {len(X)} samples")
    else:
        X, y = [], []
        print("Starting new dataset")

    # --------- INTENTS ----------
    INTENTS = {
        '0': ('Neutral', 0),
        '1': ('Reach Forward', 1),
        '2': ('Reach Up', 2),
        '3': ('Reach Side', 3),
        '4': ('Sit Down Start', 4),
        '5': ('Stand Up Start', 5),
        '6': ('Walk Start', 6),
        '7': ('Turn Left', 7),
        '8': ('Turn Right', 8),
        '9': ('Lean Forward', 9),
        'a': ('Lean Backward', 10),
        'b': ('Loss of Balance', 11)
    }

    print("\n===== INTENT KEYS =====")
    print("Press R to START recording")
    for k, v in INTENTS.items():
        print(f"{k} → {v[0]}")
    print("Press Q to quit\n")

    # ---------------- LOOP ----------------
    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)
        pose = detector.getPoseVector()

        key = cv2.waitKey(1) & 0xFF

        # START RECORDING
        if key == ord('r') and not recording:
            recording = True
            sequence = []
            print("Recording started...")

        # RECORD FRAMES
        if recording and len(pose) == 99:
            sequence.append(pose)

            if len(sequence) == SEQUENCE_LENGTH:
                recording = False
                print("Recording complete (30 frames)")
                print("Press intent key to SAVE")

        # SAVE SAMPLE
        if not recording and len(sequence) == SEQUENCE_LENGTH:
            key_char = chr(key)
            if key_char in INTENTS:
                label_name, label_id = INTENTS[key_char]

                X.append(sequence.copy())
                y.append(label_id)

                # SAVE IMMEDIATELY
                np.save(X_path, np.array(X))
                np.save(y_path, np.array(y))

                print(f"Saved → {label_name}")
                print("X shape:", np.array(X).shape)
                print("y shape:", np.array(y).shape)

                sequence.clear()

        # DISPLAY STATUS
        status = "RECORDING" if recording else "WAITING"
        cv2.putText(img, f"Status: {status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if recording else (0, 0, 255), 2)

        cv2.putText(img, f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 2)

        cv2.imshow("Intent Data Collection", img)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nFINAL DATASET SIZE")
    print("X shape:", np.array(X).shape)
    print("y shape:", np.array(y).shape)


if __name__ == "__main__":
    main()