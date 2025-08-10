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
* **Classes:**
1. fish sea_food shrimp
2. fish sea_food striped_red_mullet
3. fish sea_food sea_bass
4. animal fish bass
5. fish sea_food black_sea_sprat
6. fish sea_food red_mullet
7. fish sea_food gilt_head_bream
8. fish sea_food red_sea_bream 
9. animal fish : 1096
10. fish sea_food trout
11. fish sea_food hourse_mackerel
* **Image Size:** Resized to a fixed dimension (224x224) before training.
* **Source:** *Labmentix*

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
<img width="495" height="270" alt="image" src="https://github.com/user-attachments/assets/61d69c68-6d3d-4f4b-bc98-5690afd0092e" />

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
<img width="367" height="526" alt="image" src="https://github.com/user-attachments/assets/8a8c44d6-8809-48ad-8447-91c687ae3455" />
<img width="381" height="528" alt="image" src="https://github.com/user-attachments/assets/0772a981-545a-4022-9c29-3c0919f8752c" />


---

## 📁 Project Structure

```
📦 fish-classification
│── app.py                                                    # Streamlit app script
│── Multiclass-Fish-Image-classification.ipynb                # Model training script
│── best_mobilenet_model.h5                                   # Saved best model as .h5 file
│── requirements.txt                                          # Python dependencies
│── runtime.txt                                               # Runtime environment
│── README.md                                                 # Project documentation
│── data/                                                     # Dataset folder
```

---

## 🚀 Deployment

* The best MobileNet model (`best_mobilenet_model.h5`) was deployed using **Streamlit**.
* The app is hosted and accessible via: *Streamlit Cloud*
* (click here to view the app)[https://multiclass-fish-image-classification-dal34enyafbhoz8xlzuypn.streamlit.app/]

---

## 📌 Future Improvements

* Expand dataset for more species.
* Improve generalization with data augmentation.
* Deploy a **multi-model comparison** app.
* Optimize for mobile devices.

---
