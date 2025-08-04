# 🎓 University Chatbot: A Comparative Analysis of SVM and BERT for Intent Classification

This repository presents a university-focused chatbot that classifies user queries and responds based on intent. It implements and compares two NLP techniques:
- A traditional **Support Vector Machine (SVM)** model with TF-IDF features.
- A fine-tuned **BERT** model using the Hugging Face Transformers library.

The project aims to evaluate their performance on metrics like accuracy, contextual understanding, and response relevance.

![Chatbot Banner](https://user-images.githubusercontent.com/26465325/187948211-e6c894d9-5f16-4328-9189-e137f778d91b.png)

---

## 📚 Table of Contents

- [🔍 Project Overview](#-project-overview)
- [📁 Dataset Details](#-dataset-details)
- [🧠 Models Implemented](#-models-implemented)
  - [1. SVM Model](#1-svm-model)
  - [2. BERT Model](#2-bert-model)
- [📊 Performance Comparison](#-performance-comparison)
- [📂 Repository Structure](#-repository-structure)


---

## 🔍 Project Overview

This chatbot is built to assist university students and applicants with queries related to:
- Departments and faculty
- Courses and specializations
- Campus resources
- Scholarships
- Placement cell
- Clubs and events

We experimented with:
1. **Linear SVM** using TF-IDF features (a classic approach).
2. **BERT** (Bidirectional Encoder Representations from Transformers) fine-tuned for intent classification.

---

## 📁 Dataset Details

- **File**: `intents.json`
- **Structure**: Each intent contains:
  - `tag`: Intent name (e.g., `placements`, `scholarship_query`)
  - `patterns`: Sample user inputs
  - `responses`: Relevant replies for the chatbot to return

### 📊 Dataset Statistics
- Total patterns: **983**
- Total responses: **455**

---

## 🧠 Models Implemented

### 1. SVM Model

- **Library**: `scikit-learn`
- **Classifier**: `LinearSVC`
- **Features**: Extracted using **TF-IDF Vectorizer**
- **Performance**:  
  - Final Accuracy: **79.70%**

### 2. BERT Model

- **Library**: `transformers` (Hugging Face), `torch`
- **Pretrained Model**: `bert-base-uncased`
- **Tokenizer**: BERT tokenizer with `max_length=32`
- **Training**: Fine-tuned for **100 epochs** using AdamW optimizer

#### 📈 Accuracy Over Epochs
Epoch 10 Accuracy: 60.41%
Epoch 20 Accuracy: 73.60%
Epoch 30 Accuracy: 77.66%
...
Epoch 100 Accuracy: 79.70%
---

## 📊 Performance Comparison

| #  | Query                                                             | SVM Response                                                                                      | BERT Response                                                                                       | Relevance |
|----|-------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|-----------|
| 1  | Who is the Head of Department of ECE?                             | Dr. A. Rajani                                                                                      | Dr. A. Rajani                                                                                       | ✅ High    |
| 2  | What are the CSE professors?                                      | Dr. K P Supreethi...                                                                               | Dr. K P Supreethi...                                                                                | ✅ High    |
| 3  | Scholarship amount for M.Tech SC students?                        | M.Sc courses available...                                                                          | NSP details...                                                                                      | ❌ Low     |
| 4  | Where is JNTUH located?                                           | Website info                                                                                        | Kukatpally, Hyderabad                                                                               | ✅ Medium  |
| 5  | Email address of library?                                         | Contact library help desk                                                                          | [email protected]                                                                 | ✅ High    |
| 6  | M.Tech specializations in ECE?                                    | BTech specializations...                                                                           | 25 students per specialization                                                                     | ❌ Low     |
| 7  | Faculty in Civil Engineering?                                     | Dr B Dean Kumar                                                                                     | Dr B Dean Kumar                                                                                     | ✅ High    |
| 8  | Admission process JNTUH                                           | General info                                                                                        | jntuh.ac.in > Admissions Portal                                                                    | ✅ High    |
| 9  | IDDMP course duration?                                            | 5 years                                                                                             | 5 years                                                                                             | ✅ High    |
| 10 | B.Tech programs at JNTUH?                                         | 9 departments                                                                                       | CSE, ECE, ME, CE...                                                                                 | ✅ High    |
| 11 | Are there student clubs?                                          | JNTUH has several clubs                                                                             | JNTUHCEH clubs: technical, cultural, sports                                                         | ✅ High    |
| 12 | Equipment in EEE labs?                                            | EEE labs cater to industry needs                                                                    | EEE department has power systems, electronics, and machine labs                                     | ✅ High    |
| 13 | Eligible for NSP?                                                 | Create account and apply                                                                            | NSP provides merit and category-based scholarships                                                  | ✅ High    |
| 14 | Prathibha Scholarship info?                                       | Renewal depends on academic performance                                                             | Merit-based scholarship for Telangana students                                                      | ✅ Medium  |
| 15 | Placements in CSE?                                                | Many companies visit campus                                                                         | Active placement cell with top recruiters                                                           | ✅ High    |

### 🧾 Summary
- **BERT** shows better contextual understanding.
- **SVM** gives decent results but struggles with nuances.
- **Conclusion**: For real-world chatbot use, **BERT is recommended**.

---

