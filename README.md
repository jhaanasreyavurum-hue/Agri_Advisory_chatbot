# 🌽 Maize Advisory Chatbot  
AI-Powered Disease & Pest Detection System  
Deep Learning • Computer Vision • NLP • Voice Interface  

---

## 1. Project Overview

The Maize Advisory Chatbot is an AI-based agricultural system designed to help farmers detect maize diseases and pests and receive recommendations in real time.

The system combines:
- Image-based disease detection  
- NLP-based chatbot  
- Multilingual support (English + Hindi)  
- Voice interaction  

Users can upload maize leaf images and get:
- Disease or pest name  
- Confidence score  
- Advisory solution  

---

## 2. Key Features

### Image-Based Detection
- Detects diseases: Blight, Grey Leaf Spot, Rust, Healthy  
- Detects pests: Aphids, Corn Rootworm, Fall Army Worm, Stalk Borer  
- Uses CLIP model to reject non-maize images  
- Dual ResNet-18 models for classification  

### Conversational Advisory
- Uses SentenceTransformer (all-MiniLM-L6-v2)  
- Cosine similarity-based Q&A matching  
- Knowledge base from CSV dataset  
- Auto query generation from image results  

### Multilingual & Voice
- English and Hindi interface  
- Voice input using SpeechRecognition  
- Text-to-speech using gTTS  
- Translation using deep-translator  

---

## 3. System Architecture

The system has 4 layers:

1. Presentation Layer → Streamlit UI (`chatbot.py`)  
2. Inference Layer → Image prediction (`predict.py`)  
3. Advisory Layer → NLP + dataset matching  
4. Training Layer → Model training scripts  

### Model Pipeline

1. Image → CLIP → maize check  
2. If maize → ResNet disease + pest models  
3. Highest confidence result selected  
4. Advisory generated and returned  

---

## 4. Project Structure


Agri_Advisory_chatbot/
│── chatbot.py
│── predict.py
│── train_disease.py
│── train_pest.py
│── train_maize_check.py
│── classify.py
│── dataset.csv
│── requirements.txt
│── README.md
│── models/
│── dataset/
│── images/


---

## 5. Model Details

### CLIP (Maize Verification)
- Model: openai/clip-vit-base-patch32  
- Rejects non-maize images  

### Disease Model
- ResNet-18  
- Classes: blight, grey_leaf_spot, healthy, rust  
- Loss: CrossEntropy  
- Optimizer: Adam  

### Pest Model
- ResNet-18  
- Classes: aphids, corn_rootworm, fall_army_worm, stalk_borer  

---

## 6. Dataset

### Image Data
- maize_disease/
- maize_pest/
- maize_check_balanced/

### CSV Knowledge Base
- English questions  
- English answers  
- Hindi answers  

---

## 7. Installation

### Requirements
- Python 3.8+
- pip

### Install Libraries

```bash
pip install torch torchvision
pip install streamlit
pip install transformers sentence-transformers
pip install Pillow scikit-learn pandas
pip install gTTS SpeechRecognition deep-translator
pip install pyaudio
8. Training
python train_disease.py
python train_pest.py
python train_maize_check.py
9. Run Application
streamlit run chatbot.py

Open:
http://localhost:8501

10. Usage
Image Input
Upload maize image
System detects disease/pest
Gives recommendation
Text Input
Ask maize-related questions
System returns best answer
Voice Input
Click mic
Speak question
11. Technology Stack
Streamlit (UI)
PyTorch (Deep Learning)
ResNet-18 (Models)
CLIP (Image verification)
SentenceTransformers (NLP)
gTTS (Speech output)
SpeechRecognition (Input)
Deep Translator (Language)
12. Limitations
Works only for maize crop
CPU-based (slower inference)
Dataset not included
Large models not uploaded


## Acknowledgements
- OpenAI (CLIP)
- Hugging Face
- SentenceTransformers
- PyTorch
- Streamlit
