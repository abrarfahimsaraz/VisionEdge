
import tensorflow as tf
import os

def convert_model_to_tflite(model_path, tflite_path):
    """Converts a saved Keras model to TensorFlow Lite format."""
    try:
        # Load the model
        model = tf.keras.models.load_model(model_path)

        # Convert the model to TFLite format
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Save the TFLite model
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        print(f"Model converted to TensorFlow Lite and saved at {tflite_path}")
    except Exception as e:
        print(f"Failed to convert model: {e}")


def main():
    # Directory containing saved models
    models_dir = "./models"
    tflite_dir = "./tflite_models"

    # Ensure the output directory exists
    os.makedirs(tflite_dir, exist_ok=True)

    # List all saved models in the directory
    for model_file in os.listdir(models_dir):
        if model_file.endswith(".h5"):
            model_path = os.path.join(models_dir, model_file)
            tflite_file = os.path.splitext(model_file)[0] + ".tflite"
            tflite_path = os.path.join(tflite_dir, tflite_file)

            # Convert the model to TFLite
            print(f"Converting {model_file}...")
            convert_model_to_tflite(model_path, tflite_path)

if __name__ == "__main__":
    main()
