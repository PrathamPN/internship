import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 1. Load and Preprocess MNIST
print("\n1. Loading and Preprocessing MNIST Dataset")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")
print(f"Image shape     : {X_train.shape[1:]}")
print(f"Classes         : 0-9 (digit classification)")

# Normalize pixel values from [0,255] to [0,1]
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0

# Flatten 28x28 images to 784-dimensional vectors
X_train_flat = X_train.reshape(-1, 784)
X_test_flat  = X_test.reshape(-1, 784)

# One-hot encode labels
y_train_ohe = keras.utils.to_categorical(y_train, 10)
y_test_ohe  = keras.utils.to_categorical(y_test, 10)

print("Preprocessing complete:")
print(f"  Pixel values normalized to [0, 1]")
print(f"  Images flattened: 28x28 -> 784")
print(f"  Labels one-hot encoded: 0-9 -> 10-dim vector")

# 2. Design Neural Network Architecture
print("\n2. Designing Neural Network Architecture")
model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(256, activation='relu', name='hidden_layer_1'),
    layers.Dropout(0.3, name='dropout_1'),
    layers.Dense(128, activation='relu', name='hidden_layer_2'),
    layers.Dropout(0.2, name='dropout_2'),
    layers.Dense(64, activation='relu', name='hidden_layer_3'),
    layers.Dense(10, activation='softmax', name='output_layer')
], name="MNIST_FeedForward_NN")

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model.summary()

# 3. Train the Model (Backpropagation)
print("\n3. Training the Model (Backpropagation with Adam Optimizer)")
history = model.fit(
    X_train_flat, y_train_ohe,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# 4. Evaluate the Model
print("\n4. Evaluating the Model on Test Set")
test_loss, test_acc = model.evaluate(X_test_flat, y_test_ohe, verbose=0)
print(f"Test Loss    : {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# Per-class accuracy breakdown
y_pred_classes = np.argmax(model.predict(X_test_flat, verbose=0), axis=1)
from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes,
      target_names=[str(i) for i in range(10)]))

# 5. Visualize Training/Validation Loss and Accuracy Curves
print("\n5. Saving Training Curves")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='steelblue', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', color='darkorange', linewidth=2, linestyle='--')
axes[0].set_title('Model Accuracy over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history['loss'], label='Train Loss', color='steelblue', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', color='darkorange', linewidth=2, linestyle='--')
axes[1].set_title('Model Loss over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle(f'Neural Network Training Curves (Final Test Acc: {test_acc*100:.2f}%)',
             fontsize=13)
plt.tight_layout()
plt.savefig("level3_task3_neural_network/training_curves.png", dpi=150)
plt.close()
print("Training curves saved as 'training_curves.png'")
