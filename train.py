import tensorflow as tf
from tensorflow.keras import layers, models
import os

# ==============================
# Settings
# ==============================
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15

train_dir = "dataset/Train"       # change to your train folder path
val_dir = "dataset/Validation"    # change to your validation folder path

# ==============================
# Load dataset
# ==============================
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# ==============================
# Remove 'O' class if present
# ==============================
class_names = train_ds.class_names
if "O" in class_names:
    print("⚠️ 'O' found in dataset, removing it...")
    class_names.remove("O")

print("✅ Final class names:", class_names)
print("✅ Total classes:", len(class_names))

# Map datasets to only allowed classes
def filter_o(image, label):
    class_name = class_names[label]
    return image, label

# (Optional) Normalize pixel values
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# ==============================
# Build CNN Model
# ==============================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ==============================
# Callbacks (Save Models)
# ==============================
checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
    "bestmodel.h5", monitor="val_accuracy", save_best_only=True, verbose=1
)

# ==============================
# Train Model
# ==============================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_best]
)

# Save final model
model.save("bmodel.h5")

print("🎉 Training complete!")
print("👉 Final model saved as bmodel.h5")
print("👉 Best validation accuracy model saved as bestmodel.h5")
