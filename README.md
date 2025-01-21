# VisionEdge: Cataract Detection via Smartphone 📱👁️

**VisionEdge** is a cutting-edge solution for cataract detection, combining advanced deep learning models and Edge Intelligence. By leveraging MobileNetV2 for efficient and lightweight processing, the project achieves real-time diagnostic capability on mobile devices. This innovation aims to make cataract detection accessible, cost-effective, and scalable, especially in low-resource settings.

---

## 🌟 Key Features
- **High Accuracy**: Achieves up to **99.11% test accuracy** with DenseNet121 and MobileNetV2.
- **Efficient Deployment**: Optimized TensorFlow Lite (TFLite) models for smartphone usage.
- **User-Friendly Interface**: Integrated with the SmartEye Android app for real-time detection.
- **Robust Models**: Implements CNN, ResNet50, VGG19, DenseNet121, and MobileNetV2 for performance comparison.
- **Scalable Solution**: Designed for use in rural and resource-limited areas.

---

## 🚀 Technologies Used
- **Python**: For model training and preprocessing.
- **TensorFlow**: Framework for deep learning.
- **TensorFlow Lite**: Optimized models for Edge devices.
- **Android Studio**: For app development and deployment.
- **Gradle and Java**: Backend for SmartEye app.

---

## 📂 Dataset
- **Source**: Kaggle Cataract Dataset.
- **Composition**:
  - Training Set: 560 images.
  - Validation Set: 120 images.
  - Test Set: 120 images.
- **Balance**: Equal representation of cataract-affected and healthy eyes.

---

## 🏗️ Model Architectures
### **1. CNN**
- Rescaling, Conv2D layers, and Dense layers.
- **Accuracy**: 98.22%.

### **2. ResNet50**
- Pre-trained backbone with Global Average Pooling.
- **Accuracy**: 98.65%.

### **3. VGG19**
- Pre-trained backbone with Flatten and Dense layers.
- **Accuracy**: 98.22%.

### **4. DenseNet121**
- Pre-trained backbone with Dense layers.
- **Accuracy**: 99.11%.

### **5. MobileNetV2**
- Lightweight, optimized for mobile deployment.
- **Accuracy**: 99.11%.

---

## 📊 Model Comparison
| Model        | Test Accuracy | Precision | Recall | F1-Score | TFLite Size (MB) | Execution Time (s) |
|--------------|---------------|-----------|--------|----------|-------------------|--------------------|
| **CNN**      | 98.22%        | 0.9823    | 0.9822 | 0.9822   | 1.6               | 4.19               |
| **ResNet50** | 98.65%        | 0.9866    | 0.9865 | 0.9865   | 92.79             | 5.44               |
| **VGG19**    | 98.22%        | 0.9824    | 0.9822 | 0.9822   | 79.26             | 18.74              |
| **DenseNet121** | 99.11%     | 0.9913    | 0.9911 | 0.9911   | 27.76             | 7.07               |
| **MobileNetV2** | 99.11%     | 0.9913    | 0.9911 | 0.9911   | 9.3               | 17.44              |

---

## 📱 Deployment: SmartEye Android App
1. **Why MobileNetV2?**
   - Small TFLite size: **9.3MB**.
   - High accuracy: **99.11%**.
   - Efficient execution on mobile devices.
2. **Steps**:
   - Convert MobileNetV2 to TFLite format.
   - Integrate TFLite model into the SmartEye Android app.
   - Optimize app UI for seamless user experience.

---

## 📦 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/abrarfahimsaraz/VisionEdge.git
   cd VisionEdge
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train Models:
   ```bash
   python train_model.py
   ```

4. Convert MobileNetV2 to TFLite:
   ```bash
   python convert_to_tflite.py
   ```

5. Deploy TFLite model in the SmartEye app.

---

## 📖 Usage
1. Run the training scripts for desired models.
2. Evaluate model performance.
3. Deploy the best model (MobileNetV2) to Android for real-time cataract detection.

---

## 🙌 Contribution
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

---

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
