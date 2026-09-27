# TomatoCare AI 🌿

TomatoCare AI is a computer-vision web application built with **Streamlit** and **TensorFlow / Keras** for automated tomato leaf disease classification. The model analyzes uploaded leaf photos across 9 trained categories and provides immediate predictions, confidence scores, probability distributions, and plant health management guidance.

---

## 🌟 Key Features

* **Instant Leaf Diagnosis**: Classifies tomato leaves into 9 distinct disease/healthy categories.
* **Custom CNN Backbone**: Uses a custom 3-layer Convolutional Neural Network trained on 128×128 RGB images.
* **Interactive Probability Breakdown**: Visualizes full model probability distribution for each prediction via Plotly charts.
* **Management Guidance**: Displays concise disease descriptions, symptoms, and actionable management steps.
* **Modern UI/UX**: Built with a clean dark theme, custom status badges, and sample leaf preview galleries.

---

## 📁 Project Structure

```text
ML project streamlit/
├── model/
│   ├── tomato_disease_cnn.h5      # Trained TensorFlow/Keras model (39.7 MB)
│   └── image_model.ipynb          # Model training & evaluation notebook
├── project/
│   ├── app.py                     # Main Streamlit web application
│   ├── requirements.txt           # Python dependencies
│   ├── images/                    # Sample leaf images for UI preview
│   └── .streamlit/                # Streamlit configuration
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

---

## 🏷️ Supported Leaf Categories (9 Classes)

1. `Bacterial Spot` (`Tomato___Bacterial_spot`)
2. `Early Blight` (`Tomato___Early_blight`)
3. `Late Blight` (`Tomato___Late_blight`)
4. `Leaf Mold` (`Tomato___Leaf_Mold`)
5. `Septoria Leaf Spot` (`Tomato___Septoria_leaf_spot`)
6. `Two-Spotted Spider Mites` (`Tomato___Spider_mites Two-spotted_spider_mite`)
7. `Tomato Yellow Leaf Curl Virus` (`Tomato___Tomato_Yellow_Leaf_Curl_Virus`)
8. `Tomato Mosaic Virus` (`Tomato___Tomato_mosaic_virus`)
9. `Healthy` (`Tomato___healthy`)

---

## 📊 Model Specifications & Metrics

| Parameter | Value |
| :--- | :--- |
| **Architecture** | Custom 3-layer Sequential CNN (`Conv2D` → `MaxPooling2D` → `Dense`) |
| **Input Resolution** | `128 × 128 × 3` RGB |
| **Model Accuracy** | **91%** |
| **Precision** | **92%** |
| **Recall** | **91%** |
| **F1-Score** | **92%** |
| **Evaluation Set** | 2,112 test set leaf images |

---

## 🚀 Local Installation & Setup

### Prerequisites
* **Python 3.10+** (Tested on Python 3.11 / 3.13)
* **Git**

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/tomatocare-ai.git
   cd "ML project streamlit/project"
   ```

2. **Create and activate a virtual environment:**
   * **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   * **macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Ensure model file location:**
   Verify `tomato_disease_cnn.h5` is located inside the `model/` folder.

5. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.

---

## ☁️ Deploying to Streamlit Cloud

1. Push your repository to **GitHub**.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and log in with GitHub.
3. Click **New App**, select your repo, set **Main file path** to `project/app.py`, and click **Deploy**!

---

## 👥 Authors & Acknowledgments

* Developed by **Group 9**
* Built with Streamlit, TensorFlow, Keras, NumPy, Pillow, and Plotly.