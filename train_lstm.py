import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. LOAD DATA
# -----------------------------
X = np.load("data/X.npy")
y = np.load("data/y.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

NUM_CLASSES = len(np.unique(y))
print("Number of classes:", NUM_CLASSES)

# One-hot encode labels
y_cat = to_categorical(y, NUM_CLASSES)

# -----------------------------
# 2. TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# -----------------------------
# 3. BUILD LSTM MODEL
# -----------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 99)),
    Dropout(0.3),

    LSTM(32),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# 4. CALLBACKS
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "intent_lstm_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# -----------------------------
# 5. TRAIN MODEL
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint]
)

# -----------------------------
# 6. EVALUATE MODEL
# -----------------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc * 100:.2f}%")

# -----------------------------
# 7. CONFUSION MATRIX
# -----------------------------
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# -----------------------------
# 8. CLASSIFICATION REPORT
# -----------------------------
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, zero_division=0))

# -----------------------------
# 9. TRAINING CURVES
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()