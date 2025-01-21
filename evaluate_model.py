
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model_path, test_dataset):
    """Evaluates a model on the test dataset and generates metrics."""
    try:
        # Load the trained model
        model = tf.keras.models.load_model(model_path)

        # Evaluate on the test dataset
        print("Evaluating model...")
        test_loss, test_accuracy = model.evaluate(test_dataset)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        # Generate predictions
        y_true = []
        y_pred = []
        for images, labels in test_dataset:
            predictions = model.predict(images)
            predicted_classes = np.argmax(predictions, axis=1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted_classes)

        # Generate classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=test_dataset.class_names))

        # Plot confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.class_names, yticklabels=test_dataset.class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

    except Exception as e:
        print(f"Failed to evaluate model: {e}")


def main():
    model_dir = "./models"
    test_dataset_dir = "./data/test"
    img_size = (224, 224)
    batch_size = 32

    # Load the test dataset
    print("Loading test dataset...")
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dataset_dir,
        image_size=img_size,
        batch_size=batch_size
    )

    # Evaluate all models in the directory
    for model_file in os.listdir(model_dir):
        if model_file.endswith(".h5"):
            model_path = os.path.join(model_dir, model_file)
            print(f"\nEvaluating {model_file}...")
            evaluate_model(model_path, test_dataset)

if __name__ == "__main__":
    main()
