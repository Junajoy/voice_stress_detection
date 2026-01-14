
# Voice Stress Detection using Deep Learning

## Project Overview

This project focuses on **detecting stress levels from human speech** using deep learning techniques.
A Convolutional Neural Network (CNN) is trained on emotional speech data and deployed through a **Streamlit web application**.

The application allows users to:

* Upload pre-recorded audio files
* Record live audio using a microphone
* Analyze stress levels with confidence scores
* Visualize stress trends over time
* Compare stress levels across multiple audio samples

The system classifies speech into **Low**, **Medium**, or **High Stress** categories and provides **interpretable audio-based insights** rather than black-box predictions.

---

## Objectives

* Build a speech-based stress classification system
* Convert emotional speech labels into meaningful stress levels
* Address class imbalance using augmentation and balancing techniques
* Compare multiple deep learning models
* Deploy the final model in a local Streamlit application
* Provide explainable visual outputs for stress predictions

---

## Dataset Information

* **Dataset**: CREMA-D (Crowd-Sourced Emotional Multimodal Actors Dataset)
* **Modality Used**: Audio
* **Audio Format**: WAV
* **Speakers**: Multiple actors with varied accents and speaking styles

### CREMA-D Label Components

Each CREMA-D audio file contains:

* **Emotion**: ANG, HAP, SAD, FEAR, DIS, NEU
* **Intensity**:

  * `LO` – Low
  * `MD` – Medium
  * `HI` – High
  * `XX` – Undefined / Neutral intensity


---

## Stress Label Mapping Strategy

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

## Preprocessing Pipeline

* Audio resampling to a consistent sample rate
* Silence trimming
* MFCC extraction:

  * 40 coefficients
  * Fixed temporal length (padding / truncation)
* Feature normalization
* Stress label encoding

---

## Model Architecture

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

## Model Variants & Comparison

Three models were trained and evaluated.

### Model 1 – Baseline

* No data augmentation
* No class balancing
* Lower dropout
* Overfitting observed
* Bias toward majority stress class

---

### Model 2 – Augmentation Only

* Audio augmentation:

  * Time stretching
  * Pitch shifting
  * Noise injection
* Class imbalance still present
* Improved robustness but unstable validation performance

---

### Model 3 – Augmentation + Class Balancing (Final Model)

**Final model used in the Streamlit app**

* Audio augmentation
* Explicit class balancing
* Increased dropout in deeper layers
* Stable training curves
* Better generalization across stress classes

### Key Difference Across Models

The **primary difference between Model 1, Model 2, and Model 3** is the **use of augmentation and class balancing techniques**, directly impacting generalization and stress-class stability.

---

## Training Evaluation & Design Justification

### Architectural Choices

* Deeper layers added only after augmentation stabilized training
* Dropout rates increased after observing overfitting in Model 1
* Filter depth increased gradually to balance expressiveness and stability

### Hyperparameter Justification

* Learning rate selected based on validation loss behavior
* Dropout values justified using training history trends
* Batch size chosen for convergence stability

---

### Results & Performance Metrics
The final CNN model (Model 3 – speaker-independent split with class-weighted training) was evaluated on the validation set. Key performance metrics are summarized below:
**Overall Performance**
Metric	Value
Accuracy	0.740
Precision	0.835
Recall	0.740
F1-score	0.759

**Class-wise Accuracy**
Low Stress:	0.842
Medium Stress:	0.696
High Stress:	0.917

**Observations**
*Model 3 achieves the best generalization compared to earlier models due to the speaker-independent data split and class-weighted loss.
*Medium stress is predicted with high precision, while low-stress examples have higher recall.
*Confusion matrices and learning curves demonstrate stable training with limited overfitting.
*The macro-average F1-score of 0.70 indicates balanced performance across classes, supporting the model's suitability for deployment.

## Streamlit Application Features

* Upload WAV / MP3 audio files
* Live microphone recording

 ![the app](https://github.com/user-attachments/assets/90bc7a2c-8e9c-4b65-a547-b11a4e6d3361)
* Stress level prediction with confidence score
* Stress-over-time curve (continuous values, not discrete labels)
* Feature indicators:

  * Pitch variability
  * Spectral centroid
  * MFCC variance
* Export predictions as CSV or JSON

![image1](https://github.com/user-attachments/assets/5eec9207-6a77-43d1-a2ca-f0cf515ca6ce)

![image2](https://github.com/user-attachments/assets/e7480c00-bcae-4b2c-ba0a-0d450004732a)

![image3](https://github.com/user-attachments/assets/23dee2e7-4054-4d9b-9717-fc88f1c86ca8)
* Stress comparison across multiple files


![comparison](https://github.com/user-attachments/assets/f819d4d0-33e7-4897-8f56-03c2a4d48def)

---

## Running the Application Locally

###  Clone the Repository

```
git clone https://github.com/Junajoy/voice-stress-detection.git
cd voice-stress-detection
```

###  Create a Virtual Environment (Recommended)

```
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

###  Install Dependencies

```
pip install -r requirements.txt
```

###  Run the Application

```
streamlit run app.py
```



---

##  Sample Audio Files

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


## Future Improvements

Paralinguistic Stress Cues: Incorporate voice tremor, pitch instability, and amplitude variations (e.g., jitter, shimmer, HNR) to capture involuntary stress signals.

Filled Pause & Disfluency Analysis: Detect hesitation markers such as “um”, “uh”, and prolonged pauses as indicators of cognitive stress.

Late Fusion Strategy: Combine CNN predictions with auxiliary paralinguistic scores to refine stress confidence without affecting model training.

Advanced Speech Models: Explore transformer-based encoders for richer temporal modeling.

Multimodal & Clinical Validation: Extend to multimodal inputs and validate using clinically annotated datasets.

---

## Conclusion

This project demonstrates how retaining all CREMA-D and emotions intensities combined with **augmentation and class balancing**, significantly improves speech-based stress detection.

The final system integrates a **robust CNN model** with an **explainable, locally runnable Streamlit application**, making it suitable for real-world exploratory and interview-grade evaluation scenarios.
