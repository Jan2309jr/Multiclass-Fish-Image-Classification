# 🐟 Fish Classification using Deep Learning

## 📌 Project Overview

This project focuses on **classifying fish images** into multiple categories using **deep learning models**.
The workflow involves:

1. **Training a custom CNN model** from scratch.
2. **Leveraging transfer learning** with multiple state-of-the-art pre-trained models.
3. **Evaluating performance** and selecting the best model.
4. **Saving trained models** for future inference.
5. **Deploying the best model** (MobileNet) using **Streamlit** to create an interactive web app for predictions from user-uploaded images.

---

## 📂 Dataset

* **Type:** Image dataset containing multiple fish species.
* **Classes:** Multiple fish categories (exact list depends on dataset used).
* **Image Size:** Resized to a fixed dimension (e.g., 224x224) before training.
* **Source:** *(Mention source here if public, e.g., Kaggle, Custom dataset)*

---

## 🧠 Models Used

| Model              | Type              | Notes                              |
| ------------------ | ----------------- | ---------------------------------- |
| **Custom CNN**     | From scratch      | Baseline deep learning model.      |
| **ResNet50**       | Transfer Learning | Fine-tuned on fish dataset.        |
| **VGG16**          | Transfer Learning | Fine-tuned on fish dataset.        |
| **MobileNet**      | Transfer Learning | Achieved **best accuracy (≈99%)**. |
| **EfficientNetB0** | Transfer Learning | Balanced performance and speed.    |
| **InceptionV3**    | Transfer Learning | High accuracy and robustness.      |

---

## 📊 Model Performance

| Model          | Training Accuracy | Validation Accuracy | Test Accuracy |
| -------------- | ----------------- | ------------------- | ------------- |
| Custom CNN     | \~XX%             | \~XX%               | \~XX%         |
| ResNet50       | \~XX%             | \~XX%               | \~XX%         |
| VGG16          | \~XX%             | \~XX%               | \~XX%         |
| MobileNet      | \~99%             | \~99%               | \~99%         |
| EfficientNetB0 | \~XX%             | \~XX%               | \~XX%         |
| InceptionV3    | \~XX%             | \~XX%               | \~XX%         |

> **Note:** MobileNet was selected for deployment due to its lightweight architecture, high accuracy, and fast inference speed.

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Deep Learning:** TensorFlow / Keras
* **Data Handling:** NumPy, Pandas
* **Image Processing:** OpenCV, Pillow
* **Visualization:** Matplotlib, Seaborn
* **Web Deployment:** Streamlit
* **Model Deployment File:** `.h5` format

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/fish-classification.git
cd fish-classification
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Download the Dataset

* Place the dataset in the `data/` folder.
* Update the dataset path in the training script.

### 4️⃣ Train Models (Optional)

```bash
python train.py
```

### 5️⃣ Run the Streamlit Web App

```bash
streamlit run app.py
```

---

## 🖥️ Web App Features

✅ Upload an image of a fish.
✅ Model predicts fish species with **confidence scores**.
✅ Confidence scores are shown as a **bar graph**.
✅ Mobile-friendly interface.

---

## 📁 Project Structure

```
📦 fish-classification
│── app.py                  # Streamlit app script
│── train.py                # Model training script
│── best_mobilenet_model.h5 # Saved best model
│── requirements.txt        # Python dependencies
│── README.md               # Project documentation
│── data/                   # Dataset folder
│── models/                 # Saved model weights
│── utils.py                # Helper functions
```

---

## 🚀 Deployment

* The best MobileNet model (`best_mobilenet_model.h5`) was deployed using **Streamlit**.
* The app is hosted and accessible via: *(Add Hugging Face / Streamlit Cloud / other link here)*

---

## 📌 Future Improvements

* Expand dataset for more species.
* Improve generalization with data augmentation.
* Deploy a **multi-model comparison** app.
* Optimize for mobile devices.

---

## 🏆 Results Summary

* **Best Model:** MobileNet
* **Accuracy:** \~99% on test set
* **Inference Speed:** <1 sec per image on CPU
* **Deployment:** Streamlit web app

---

## 📜 License

This project is licensed under the MIT License. Feel free to use and modify for research or educational purposes.

---
