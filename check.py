import numpy as np

X = np.load("data/X.npy")
y = np.load("data/y.npy")

print(X)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("First 10 labels:", y[:360:10])