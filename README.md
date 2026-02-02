# 🧠 Human Intent Prediction using Pose-Based Temporal Analysis 🎯

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-MediaPipe-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📌 Project Overview 🚀
This project implements a **real-time human intent prediction system** that identifies what action a person is *about to perform* **before the action is fully completed**, using **temporal analysis of body pose data** from live video streams.

Unlike traditional action recognition systems that react *after* an action finishes, this system focuses on **early intent prediction** by learning subtle preparatory movements such as posture shifts, balance changes, and limb motion.

---

## 🎯 Why This Project Matters
✅ Early decision-making  
✅ Proactive human–machine interaction  
✅ Lightweight & real-time  
✅ No GPU dependency  

**Use cases include:**
- 🤖 Human–Robot Interaction  
- 🧓 Fall prevention systems  
- 🏃 Sports performance analysis  
- 🕶️ Smart surveillance & AR/VR systems  

---

## 🧠 System Workflow 🛠️

Live Video 🎥
↓
Pose Estimation (MediaPipe)
↓
Pose Sequence Construction
↓
LSTM Temporal Modeling
↓
Early Human Intent Prediction ✅


📌 *Add a system architecture diagram here:*  
```markdown
![System Architecture](images/system_architecture.png)
🏗️ Methodology & Pipeline 🔍
1️⃣ Pose Estimation 🧍
Webcam captures live video frames

MediaPipe Pose extracts 33 full-body landmarks per frame

Each landmark provides (x, y, z) coordinates

📌 Example pose visualization:

![Pose Estimation](images/pose_estimation.png)
2️⃣ Temporal Sequence Formation ⏱️
Pose landmarks flattened into 99-dimensional vectors

Fixed-length sequences of 30 frames

Each sequence captures early motion cues

3️⃣ Intent Prediction Model 🧠
LSTM (Long Short-Term Memory) network

Learns temporal motion patterns

Predicts intent before full action execution

🧪 Dataset Details 📊
Custom dataset collected using live pose extraction

12 human intent classes

Each sample shape: (30 × 99)

Stored in NumPy format:

data/X.npy → Pose sequences

data/y.npy → Intent labels

📌 Add dataset visualization here:

![Dataset Overview](images/dataset.png)
⚙️ Model Architecture 🧩
Input: Pose sequences (30, 99)

2 × LSTM layers for temporal learning

Dropout layers to reduce overfitting

Dense layers for classification

Loss: Categorical Cross-Entropy

🚀 Key Features ✨
✅ Real-time intent prediction
✅ Lightweight CPU-based execution
✅ Modular and extensible design
✅ Clean separation of data collection, training, and inference

📁 Project Structure 📂
Human_Intent_Prediction/
│
├── data/
│   ├── X.npy
│   └── y.npy
│
├── pose_sequence_collector.py   🎥 Pose data collection
├── build_lstm.py                🧠 Model architecture
├── train_lstm.py                📈 Model training
├── body.py                      🧍 Body pose utilities
├── hand.py                      ✋ Hand pose utilities
├── check.py                     ✅ Validation helpers
├── .gitignore
└── README.md
🛠️ Tech Stack 🧰
Language: Python 🐍

Computer Vision: MediaPipe, OpenCV 👁️

Deep Learning: TensorFlow (LSTM) 🔥

Data Handling: NumPy 📊

Execution: CPU-based ⚡

▶️ How to Run ▶️
1️⃣ Clone the repository
git clone https://github.com/Krishnakoushik21/Human_Intent_Prediction.git
cd Human_Intent_Prediction
2️⃣ Create & activate virtual environment
python -m venv mp_env
mp_env\Scripts\activate   # Windows
3️⃣ Install dependencies
pip install tensorflow mediapipe opencv-python numpy
4️⃣ Collect pose data
python pose_sequence_collector.py
5️⃣ Train the model
python train_lstm.py
📈 Results & Observations 📊
Reliable intent prediction across 12 classes

Temporal modeling outperformed single-frame pose analysis

Real-time inference achieved without GPU acceleration

🔮 Future Enhancements 🚧
🔹 Object-level context integration
🔹 Multi-person intent prediction
🔹 Larger and diverse datasets
🔹 Attention-based temporal models


