# face_emotion.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Dataset Paths
# -----------------------------
DATASET_DIR = "fer2013"
train_dir = os.path.join(DATASET_DIR, "train")
test_dir = os.path.join(DATASET_DIR, "test")

# -----------------------------
# 2️⃣ Data Generators with Preprocessing for MobileNetV2
# -----------------------------
### UPDATED ###
# Pre-trained models require a specific input size and 3 color channels (RGB).
img_size = (96, 96)

# Use the preprocessing function specific to the chosen model (MobileNetV2)
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    preprocessing_function=mobilenet_preprocess # Use MobileNetV2's preprocessing
)

# Test generator only needs the preprocessing function
test_datagen = ImageDataGenerator(
    preprocessing_function=mobilenet_preprocess # Use MobileNetV2's preprocessing
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    color_mode="rgb",  # ### UPDATED ### Pre-trained models need 3 channels
    batch_size=64,
    class_mode="categorical",
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    color_mode="rgb",  # ### UPDATED ### Pre-trained models need 3 channels
    batch_size=64,
    class_mode="categorical",
    shuffle=False
)

# -----------------------------
# 3️⃣ Compute Class Weights
# -----------------------------
from sklearn.utils.class_weight import compute_class_weight

y_train = train_generator.classes
class_weights_raw = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: w for i, w in enumerate(class_weights_raw)}
print("✅ Class indices:", train_generator.class_indices)
print("✅ Class weights:", class_weights)


# -----------------------------
# 4️⃣ ### NEW ### Build Model with Transfer Learning
# -----------------------------
# Load the pre-trained MobileNetV2, excluding its final classifier layer
base_model = MobileNetV2(
    input_shape=(img_size[0], img_size[1], 3),
    include_top=False,  # Exclude the final Dense layer
    weights='imagenet'
)

# Freeze the layers of the base model so we don't ruin their learned weights
base_model.trainable = False

# Add our custom classifier on top of the base model
inputs = Input(shape=(img_size[0], img_size[1], 3))
x = base_model(inputs, training=False) # Set training=False for the frozen base
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=inputs, outputs=predictions)

# Compile the model for the first stage of training
optimizer = Adam(1e-3) # Use a higher learning rate initially
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# -----------------------------
# 5️⃣ Callbacks
# -----------------------------
callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint("face_emotion_best_transfer.keras", save_best_only=True, monitor="val_accuracy", mode="max", verbose=1)
]


# -----------------------------
# 6️⃣ ### NEW ### Two-Stage Training (Train Head -> Fine-Tune)
# -----------------------------
print("\n--- STAGE 1: TRAINING THE CLASSIFIER HEAD ---")
INITIAL_EPOCHS = 25
history = model.fit(
    train_generator,
    epochs=INITIAL_EPOCHS,
    validation_data=test_generator,
    class_weight=class_weights,
    callbacks=callbacks
)

print("\n--- STAGE 2: FINE-TUNING THE MODEL ---")
# Unfreeze the top layers of the base model for fine-tuning
base_model.trainable = True

# We'll fine-tune from the 100th layer onwards. The earlier layers learn
# very generic features, which we don't want to change much.
for layer in base_model.layers[:100]:
    layer.trainable = False

# Re-compile the model with a very low learning rate for fine-tuning
optimizer_fine_tune = Adam(1e-5) # Crucial to use a low LR
model.compile(optimizer=optimizer_fine_tune, loss='categorical_crossentropy', metrics=['accuracy'])
print("✅ Model re-compiled for fine-tuning.")
model.summary() # Note the increase in trainable parameters

# Continue training from where we left off
FINE_TUNE_EPOCHS = 25
total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

history_fine_tune = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1], # Start from where the last stage ended
    validation_data=test_generator,
    class_weight=class_weights,
    callbacks=callbacks
)


# -----------------------------
# 7️⃣ Save Final Model
# -----------------------------
model.save("face_emotion_model_final.keras")
print("🎉 Final model saved as face_emotion_model_final.keras")


# -----------------------------
# 8️⃣ ### UPDATED ### Plot Full Training Curves
# -----------------------------
# Combine the history from both training stages
acc = history.history['accuracy'] + history_fine_tune.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine_tune.history['val_accuracy']
loss = history.history['loss'] + history_fine_tune.history['loss']
val_loss = history.history['val_loss'] + history_fine_tune.history['val_loss']

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
# Add a vertical line to show where fine-tuning started
plt.axvline(INITIAL_EPOCHS-1, color='r', linestyle='--', label='Start Fine-Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('epoch')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
# Add a vertical line to show where fine-tuning started
plt.axvline(INITIAL_EPOCHS-1, color='r', linestyle='--', label='Start Fine-Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')

plt.tight_layout()
plt.show()