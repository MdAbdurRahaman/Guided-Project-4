# 🐔 Chicken Disease Classification using Deep Learning & MLOps Pipeline (DVC)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Pipeline-violet.svg)](https://dvc.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red.svg)](https://guided-project-4-nberto6ipimmgoun5vzyq9.streamlit.app/)
[![Flask](https://img.shields.io/badge/Flask-Web%20UI-green.svg)](https://flask.palletsprojects.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

An end-to-end, production-ready Deep Learning & MLOps application built to classify chicken fecal images as **Coccidiosis** or **Healthy**. This repository demonstrates a complete machine learning lifecycle—from dataset ingestion and transfer learning model training to Data Version Control (**DVC**) pipeline tracking, Streamlit interactive app, Flask REST web service, Docker containerization, and GitHub Actions CI/CD automation.

> 🌐 **Live Web Application Demo**: [https://guided-project-4-nberto6ipimmgoun5vzyq9.streamlit.app/](https://guided-project-4-nberto6ipimmgoun5vzyq9.streamlit.app/)

---

## 📌 Problem Statement

Coccidiosis is a contagious, high-mortality parasitic disease affecting poultry worldwide. Early and accurate detection through fecal image analysis is crucial for farm biosecurity and reducing financial loss. 

This project delivers an automated visual diagnostic solution powered by a fine-tuned **VGG16 Convolutional Neural Network**, encapsulated within a modular MLOps workflow to ensure reproducibility, versioning, and seamless cloud deployment.

---

## 🌐 Live Application & Demo

Experience the interactive Streamlit diagnostic platform directly in your web browser:  
👉 **[Open Live App: PoultryHealth AI Streamlit Cloud](https://guided-project-4-nberto6ipimmgoun5vzyq9.streamlit.app/)**

---

## 🏗️ System Architecture & MLOps Pipeline

```
                              ┌────────────────────────┐
                              │  Fecal Image Dataset   │
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │ Stage 01: Data         │
                              │ Ingestion & Extraction │
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │ Stage 02: Prepare Base │
                              │ Model (VGG16 Transfer) │
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │ Stage 03: Model        │
                              │ Training & Checkpoints │
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │ Stage 04: Model        │
                              │ Evaluation & Metrics   │
                              └───────────┬────────────┘
                                          │
                                          ▼
┌────────────────────────┐    ┌────────────────────────┐
│  Docker Container      │◄───┤  Streamlit & Flask Apps│
│  & AWS/Azure Deploy    │    │ (Diagnosis & Feedback) │
└────────────────────────┘    └────────────────────────┘
```

---

## 🛠️ Tech Stack & Key Tools

- **Deep Learning**: TensorFlow / Keras (VGG16 Architecture)
- **MLOps & Versioning**: DVC (Data Version Control)
- **Deployment & UI**: Streamlit Cloud & Flask REST UI
- **Package Management**: Setuptools, Python-Box, PyYAML
- **Containerization**: Docker
- **CI/CD Pipeline**: GitHub Actions & AWS ECR / EC2

---

## 📁 Repository Structure

```
├── .github/workflows/       # GitHub Actions CI/CD pipeline definitions
│   └── main.yaml
├── config/                  # Configuration YAML paths
│   └── config.yaml
├── research/                # Notebook experiments & trials
│   └── trials.ipynb
├── src/cnnClassifier/       # Modular Python package
│   ├── components/          # Pipeline component implementations
│   │   ├── data_ingestion.py
│   │   ├── prepare_base_model.py
│   │   ├── prepare_callbacks.py
│   │   ├── training.py
│   │   └── evaluation.py
│   ├── config/              # Configuration manager
│   │   └── configuration.py
│   ├── entity/              # Dataclass entities
│   │   └── config_entity.py
│   ├── pipeline/            # Stage execution drivers
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_prepare_base_model.py
│   │   ├── stage_03_training.py
│   │   ├── stage_04_evaluation.py
│   │   └── predict.py
│   └── utils/               # Common helper utilities
│       └── common.py
├── templates/               # Web UI template
│   └── index.html
├── app.py                   # Flask server application entry point
├── streamlit_app.py         # Streamlit interactive application
├── Dockerfile               # Containerization specification
├── dvc.yaml                 # DVC pipeline stages tracking definition
├── main.py                  # Orchestration script for all pipeline stages
├── params.yaml              # Hyperparameters (epochs, batch size, learning rate)
├── requirements.txt         # Project dependencies
├── setup.py                 # Local package setup script
└── template.py              # Directory generator automation script
```

---

## 🚀 Quick Start & Deployment

### 1. Live Cloud Web App
Open [https://guided-project-4-nberto6ipimmgoun5vzyq9.streamlit.app/](https://guided-project-4-nberto6ipimmgoun5vzyq9.streamlit.app/) directly.

### 2. Clone Repository & Setup Local Environment
```bash
git clone https://github.com/MdAbdurRahaman/Guided-Project-4.git
cd Guided-Project-4
```

### 3. Create Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Execute Pipeline with DVC
```bash
# Run all stages defined in dvc.yaml
dvc repro
```
*Or run all stages manually:*
```bash
python main.py
```

### 6. Launch Streamlit Application
```bash
streamlit run streamlit_app.py
```

### 7. Launch Flask Web Application
```bash
python app.py
```

---

## 🐳 Docker Containerization

To build and run the Docker image locally:
```bash
docker build -t chicken-classifier .
docker run -p 8080:8080 chicken-classifier
```

---

## 📊 Pipeline Stages Explained

1. **Stage 01 - Data Ingestion**: Downloads the poultry fecal dataset and unzips raw image artifacts.
2. **Stage 02 - Base Model Preparation**: Downloads pre-trained VGG16 weights, freezes convolutional base layers, and attaches custom classification head.
3. **Stage 03 - Model Training**: Applies image augmentation (rotation, shearing, zoom, horizontal flip) and trains the model with checkpoint callbacks.
4. **Stage 04 - Evaluation**: Evaluates validation accuracy/loss and generates `scores.json`.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
