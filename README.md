
# Automated Evaluation of Descriptive Answer Sheets Using OCR and Semantic NLP

## 📌 Overview

Automated Evaluation of Descriptive Answer Sheets Using OCR and Semantic NLP is an AI-powered educational technology solution that automates the assessment of handwritten descriptive answers. The system combines Optical Character Recognition (OCR) and Natural Language Processing (NLP) techniques to extract handwritten text, analyze semantic meaning, and generate objective evaluation scores.

Traditional descriptive answer evaluation is time-consuming, inconsistent, and susceptible to human bias. This project addresses these challenges by providing a scalable and intelligent grading system capable of assessing handwritten responses efficiently and accurately through semantic similarity analysis.

---

## 🎯 Problem Statement

Educational institutions spend significant time and effort manually evaluating descriptive answer sheets. Manual grading often results in:

- Subjective evaluation
- Human bias and inconsistency
- Delayed result processing
- Increased workload for educators
- Difficulty in scaling assessments

This project aims to automate the evaluation process while maintaining fairness, consistency, and efficiency.

---

## 🚀 Key Features

### 📝 Handwritten Text Extraction
- OCR-based answer sheet processing
- Adaptive image preprocessing
- Handwritten text recognition using Tesseract OCR

### 🧠 Semantic Answer Evaluation
- Sentence Transformer-based embeddings
- Context-aware answer comparison
- Semantic similarity scoring
- Support for paraphrased responses

### 📊 Automated Grading
- Cosine similarity-based evaluation
- Objective and consistent scoring
- Reduced human intervention

### ✅ OCR Quality Validation
- OCR confidence assessment
- Detection of low-quality text extraction
- Improved evaluation reliability

### ⚡ FastAPI Backend
- RESTful API architecture
- Scalable and modular design
- Efficient request handling

### 🌐 Interactive Web Interface
- User-friendly dashboard
- Answer upload and evaluation workflow
- Score visualization and feedback generation

---

## 🏗️ System Architecture

```text
Student Answer Sheet
          │
          ▼
 Image Preprocessing
          │
          ▼
     OCR Engine
   (Tesseract OCR)
          │
          ▼
   Extracted Text
          │
          ▼
  Semantic Analysis
(Sentence Transformers)
          │
          ▼
 Similarity Computation
  (Cosine Similarity)
          │
          ▼
    Score Generation
          │
          ▼
  Feedback & Results
````

---

## 🛠️ Technology Stack

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* Python
* FastAPI

### AI / Machine Learning

* Sentence Transformers
* BERT Embeddings
* Semantic Similarity Analysis

### OCR

* Tesseract OCR
* OpenCV

### Database

* SQLite

---

## 📂 Project Structure

```text
Automated-Evaluation-of-Descriptive-Answer-Sheets-Using-OCR-and-Semantic-NLP
│
├── frontend/
│   ├── static/
│   ├── templates/
│   └── assets/
│
├── backend/
│   ├── api/
│   ├── models/
│   ├── services/
│   └── utils/
│
├── uploads/
├── outputs/
├── models/
├── requirements.txt
├── main.py
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Adhisheshu1210/Automated-Evaluation-of-Descriptive-Answer-Sheets-Using-OCR-and-Semantic-NLP.git

cd Automated-Evaluation-of-Descriptive-Answer-Sheets-Using-OCR-and-Semantic-NLP
```

### 2. Create Virtual Environment

python -m venv venv


### Windows

venv\Scripts\activate


### Linux / macOS


source venv/bin/activate

### 3. Install Dependencies

pip install -r requirements.txt


### 4. Install Tesseract OCR

Download and install Tesseract OCR:

https://github.com/tesseract-ocr/tesseract

---

## ▶️ Running the Application

Start the FastAPI server:

uvicorn main:app --reload


Application URL:

http://127.0.0.1:8000


API Documentation:

http://127.0.0.1:8000/docs


---

## 📖 Workflow

### Step 1

Upload scanned handwritten answer sheets.

### Step 2

Perform image preprocessing to improve OCR accuracy.

### Step 3

Extract textual content using OCR.

### Step 4

Generate semantic embeddings for extracted answers and model answers.

### Step 5

Compute similarity scores using cosine similarity.

### Step 6

Generate evaluation results and feedback.

---

## 📈 Benefits

* Faster evaluation process
* Reduced faculty workload
* Consistent grading methodology
* Improved assessment transparency
* Scalable for large examinations
* Enhanced educational analytics

---

## 🔮 Future Enhancements

* Multi-language answer evaluation
* LLM-powered grading and feedback
* Subject-specific evaluation rubrics
* Institution-level analytics dashboard
* Cloud deployment support
* LMS integration
* Vision-Language Model support

---

## 🧪 Testing

The system has been tested for:

* OCR text extraction
* Semantic similarity calculation
* Answer scoring accuracy
* API functionality
* File upload validation
* OCR confidence detection

---

## 🔒 Security Considerations

* Input validation for uploaded files
* Secure API request handling
* Protection against malformed inputs
* Separation of uploaded content and application logic

---

## 👥 Contributors

* Angothu Adhisheshu
* Sathvik Rudam
* Nikhitha Rayindla

---

## 🎥 Demo Video

https://www.youtube.com/watch?v=JCERvxT2qTk

---

## 📷 Sample Use Cases

* University examinations
* Subjective answer evaluation
* Online assessment systems
* Educational institutions
* Coaching centers
* Academic research

---

## 📄 License

This project is developed for educational, research, and demonstration purposes.

Feel free to use, modify, and extend the project with proper attribution.

---

## ⭐ Support

If you found this project useful:

* Star the repository
* Fork the project
* Share feedback and suggestions
* Contribute improvements

---

### Building smarter, fairer, and faster educational assessments using AI, OCR, and Semantic NLP.





