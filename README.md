# ♻️ EcoVision AI — Smart Waste Classification & Disposal Guide

EcoVision AI is an AI-powered web application that identifies common waste categories from uploaded images and provides simple, step-by-step disposal guidance.

Built using **PyTorch, Transfer Learning, and Streamlit**, the project aims to make responsible waste disposal easier and more accessible.

# 🌱 Features

- **Image-Based Classification:** Identifies six waste categories from an uploaded image.
- **Step-by-Step Disposal Guidance:** Provides practical disposal instructions based on the predicted category.
- **Simple Web Interface:** Easy-to-use application built with Streamlit.
- **Deep Learning:** Uses a fine-tuned ResNet18 model with transfer learning.

# 🗂️ Waste Categories

The model classifies images into six categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

# 🛠️ Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch, Torchvision
- **Model:** ResNet18 — Transfer Learning
- **Image Processing:** Pillow
- **Web Application:** Streamlit
- **Environment:** Google Colab and PyCharm

# 📊 Model Performance

The model was evaluated on a held-out test dataset.

| Metric | Result |
|---|---:|
| Test Accuracy | 88.68% |
| Macro F1-Score | 87.58% |
| Weighted F1-Score | 88.66% |

The results reflect performance on the test dataset. Predictions on new images may vary.

# 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/EcoVision-AI.git
cd EcoVision-AI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python -m streamlit run EcoVision_AI_streamlit.py
```

Open the local URL displayed in your terminal to use the application.

# 📁 Project Structure

```text
EcoVision-AI/
├── EcoVision_AI_streamlit.py
├── requirements.txt
├── README.md
└── EcoVision_AI.ipynb
```

# ⚠️ Limitations

- The model predicts one of six predefined waste categories.
- Incorrect predictions are possible, particularly for unfamiliar items or images containing multiple objects.
- Disposal recommendations are general guidance. Always follow local waste collection and recycling rules.

# 🌍 Purpose

EcoVision AI explores how computer vision can help people make more informed waste disposal decisions.

*“Small actions today create a greener tomorrow.”*

---

**Built as an AI/ML portfolio project.**
