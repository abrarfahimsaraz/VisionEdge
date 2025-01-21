
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, DenseNet121, ResNet50, VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import os

# Load datasets
def load_datasets(base_dir, img_size, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(base_dir, "train"),
        image_size=img_size,
        batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(base_dir, "validation"),
        image_size=img_size,
        batch_size=batch_size
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(base_dir, "test"),
        image_size=img_size,
        batch_size=batch_size
    )
    return train_ds, val_ds, test_ds

# Create model factory
def create_model(model_type, input_shape, num_classes):
    if model_type == "MobileNetV2":
        base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    elif model_type == "DenseNet121":
        base_model = DenseNet121(input_shape=input_shape, include_top=False, weights="imagenet")
    elif model_type == "ResNet50":
        base_model = ResNet50(input_shape=input_shape, include_top=False, weights="imagenet")
    elif model_type == "VGG19":
        base_model = VGG19(input_shape=input_shape, include_top=False, weights="imagenet")
    else:
        raise ValueError("Unsupported model type")

    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer=Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])

    return model

# Train model
def train_model(model, train_ds, val_ds, epochs):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    return history

# Main function
def main():
    base_dir = "./data"
    img_size = (224, 224)
    batch_size = 32
    input_shape = img_size + (3,)
    num_classes = 2

    train_ds, val_ds, test_ds = load_datasets(base_dir, img_size, batch_size)

    for model_type in ["MobileNetV2", "DenseNet121", "ResNet50", "VGG19"]:
        print(f"Training {model_type}...")
        model = create_model(model_type, input_shape, num_classes)
        history = train_model(model, train_ds, val_ds, epochs=10)

        # Save the model
        model.save(f"{model_type}_model.h5")

        print(f"{model_type} training completed and saved.")

if __name__ == "__main__":
    main()
