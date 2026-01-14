
# 🎙️ Voice Stress Detection using Deep Learning

## 📌 Project Overview

This project focuses on **detecting stress levels from human speech** using deep learning techniques.
A Convolutional Neural Network (CNN) is trained on emotional speech data and deployed through a **locally runnable Streamlit web application**.

The application allows users to:

* Upload pre-recorded audio files
* Record live audio using a microphone
* Analyze stress levels with confidence scores
* Visualize stress trends over time
* Compare stress levels across multiple audio samples

The system classifies speech into **Low**, **Medium**, or **High Stress** categories and provides **interpretable audio-based insights** rather than black-box predictions.

---

## 🎯 Objectives

* Build a speech-based stress classification system
* Convert emotional speech labels into meaningful stress levels
* Address class imbalance using augmentation and balancing techniques
* Compare multiple deep learning models
* Deploy the final model in a local Streamlit application
* Provide explainable visual outputs for stress predictions

---

## 📂 Dataset Information

* **Dataset**: CREMA-D (Crowd-Sourced Emotional Multimodal Actors Dataset)
* **Modality Used**: Audio
* **Audio Format**: WAV
* **Speakers**: Multiple actors with varied accents and speaking styles

### 🔖 CREMA-D Label Components

Each CREMA-D audio file contains:

* **Emotion**: ANG, HAP, SAD, FEAR, DIS, NEU
* **Intensity**:

  * `LO` – Low
  * `MD` – Medium
  * `HI` – High
  * `XX` – Undefined / Neutral intensity


---

## 🔄 Stress Label Mapping Strategy

Emotions and intensities are mapped to **stress levels** as follows:

* NEU (any intensity, including XX) → Low Stress
* HAP + LO / MD → Low Stress
* HAP + HI → Medium Stress
* SAD / DIS + LO / MD → Medium Stress
* SAD / DIS + HI → High Stress
* ANG / FEAR (any intensity) → High Stress
* Any emotion + XX → Low Stress

This ensures:

* Undefined intensity is handled gracefully
* Emotion–intensity interactions are preserved

---

## ⚙️ Preprocessing Pipeline

* Audio resampling to a consistent sample rate
* Silence trimming
* MFCC extraction:

  * 40 coefficients
  * Fixed temporal length (padding / truncation)
* Feature normalization
* Stress label encoding

---

## 🧠 Model Architecture

All models use a **CNN-based architecture on MFCC spectrograms**.

### Architecture Overview

* 2D Convolution layers
* Batch Normalization
* Max Pooling
* Dropout regularization
* Fully connected Dense layers
* Softmax output (Low / Medium / High stress)

### Why CNN + MFCC?

* MFCCs capture perceptually meaningful speech features
* CNNs learn local time–frequency patterns effectively
* Efficient for real-time inference

---

## 🧪 Model Variants & Comparison

Three models were trained and evaluated.

### 🟦 Model 1 – Baseline

* No data augmentation
* No class balancing
* Lower dropout
* Overfitting observed
* Bias toward majority stress class

---

### 🟨 Model 2 – Augmentation Only

* Audio augmentation:

  * Time stretching
  * Pitch shifting
  * Noise injection
* Class imbalance still present
* Improved robustness but unstable validation performance

---

### 🟩 Model 3 – Augmentation + Class Balancing (Final Model)

✅ **Final model used in the Streamlit app**

* Audio augmentation
* Explicit class balancing
* Increased dropout in deeper layers
* Stable training curves
* Better generalization across stress classes

### 🔑 Key Difference Across Models

The **primary difference between Model 1, Model 2, and Model 3** is the **use of augmentation and class balancing techniques**, directly impacting generalization and stress-class stability.

---

## 📉 Training Evaluation & Design Justification

### Architectural Choices

* Deeper layers added only after augmentation stabilized training
* Dropout rates increased after observing overfitting in Model 1
* Filter depth increased gradually to balance expressiveness and stability

### Hyperparameter Justification

* Learning rate selected based on validation loss behavior
* Dropout values justified using training history trends
* Batch size chosen for convergence stability

---

## 📊 Results & Performance Metrics

Evaluation metrics include:

* Accuracy
* Precision, Recall, F1-score (per class)
* Confusion matrices
* Training vs validation accuracy curves
* Training vs validation loss curves

Confusion matrices and learning curves are included to:

* Justify architectural and regularization choices
* Support final model selection
* Demonstrate improved generalization in Model 3

---

## 🖥️ Streamlit Application Features

* Upload WAV / MP3 audio files
* Live microphone recording
* Stress level prediction with confidence score
* Stress-over-time curve (continuous values, not discrete labels)
* Feature indicators:

  * Pitch variability
  * Spectral centroid
  * MFCC variance
* Stress comparison across multiple files
* Export predictions as CSV or JSON

---

## ▶️ Running the Application Locally

### 1️⃣ Clone the Repository

```
git clone https://github.com/Junajoy/voice-stress-detection.git
cd voice-stress-detection
```

### 2️⃣ Create a Virtual Environment (Recommended)

```
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```
streamlit run app.py
```

📌 The app is designed to run **locally**.
There are **no Colab, ngrok, or cloud-specific dependencies**.

---

## 🎧 Sample Audio Files

Three sample files are provided for quick testing:

```
/samples/
 ├── sample1.wav
 ├── sample2.wav
 └── sample3.wav
```

These can be used to:

* Test predictions quickly
* Demonstrate comparison charts
* Validate end-to-end inference

---

## 📸 Screenshots & Visuals

### Recommended Placement in README

* App UI overview → After “Streamlit Application Features”
* Stress comparison graph → Results & Performance Metrics
* Confusion matrix → Results & Performance Metrics
* Accuracy / Loss curves → Training Evaluation

Suggested folder:

```
/assets/screenshots/
```

Example usage:

```
![App UI](assets/screenshots/app_ui.png)
```

---

## 🚀 Future Improvements

* Transformer-based speech encoders
* Multimodal stress detection (audio + text)
* Speaker normalization
* Continuous real-time monitoring
* Clinical-grade datasets and validation

---

## ✅ Conclusion

This project demonstrates how retaining all CREMA-D and emotions intensities combined with **augmentation and class balancing**, significantly improves speech-based stress detection.

The final system integrates a **robust CNN model** with an **explainable, locally runnable Streamlit application**, making it suitable for real-world exploratory and interview-grade evaluation scenarios.
