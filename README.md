---

# Fake Review Detection Project

This project aims to detect fake reviews using **Machine Learning techniques**. The workflow includes **data preprocessing, feature engineering, model training, evaluation, web scraping, and building a website interface** (partially implemented).

---

## 📌 Project Workflow

1. **Checkpoint 1: Data Preprocessing**
2. **Checkpoint 2: Fake Review Detection (Model Training & Evaluation)**
3. **Checkpoint 3: Web Scraping Reviews**
4. **Checkpoint 4: Website for Fake Review Detection** *(In Progress)*

---

## 🧹 Data Preprocessing (Checkpoint 1)

### Steps Involved

1. **Loading Data** – Load dataset into the environment.
2. **Data Cleaning** – Handle missing values, remove duplicates, drop irrelevant data.
3. **Text Normalization** – Convert text to lowercase, remove special characters, punctuation, and numbers.
4. **Tokenization** – Split text into words, remove stopwords.
5. **Stemming & Lemmatization** – Reduce words to their base form.
6. **Feature Representation** –

   * Bag-of-Words (BoW)
   * TF-IDF (Term Frequency-Inverse Document Frequency)
   * Word Embeddings (e.g., GloVe)

✅ The preprocessed dataset was saved into a **CSV file** for model training.

**Key Learnings**:

* Word embeddings perform better for context-based tasks.
* Preprocessing improves performance by removing noise.

---

## 🤖 Fake Review Detection (Checkpoint 2)

### Models Used

1. **Random Forest Classifier** – Ensemble method for improved accuracy.
2. **Support Vector Machine (SVM)** – Finds optimal hyperplane for classification.
3. **Logistic Regression** – Linear classifier for binary classification.

### Steps

* **Data Splitting** – Train (80%) / Test (20%).
* **Pipeline Creation** – Scaling + Model.
* **Training & Evaluation** – Accuracy, Precision, Recall, F1-Score.
* **Hyperparameter Tuning** – GridSearchCV for best parameters.
* **Model Saving** – Models saved using `joblib` for later use.

**Example Results (Random Forest):**

```
Accuracy: ~75%  
Precision: 0.73 - 0.77  
Recall: 0.70 - 0.80  
F1-Score: 0.73 - 0.76  
```

---

## 🕸️ Web Scraping Reviews (Checkpoint 3)

### Basic Components of Web Scraping

1. **Website Analysis**

   * Inspect HTML structure using Developer Tools.
   * Identify tags for reviews, ratings, reviewer names, and dates.

2. **Libraries & Tools**

   * `requests` → Fetch HTML content.
   * `BeautifulSoup` or `lxml` → Parse HTML.

3. **Scraping Reviews**

   * Extract review text from product pages.
   * Handle multi-line reviews & embedded HTML tags.

4. **Save Data**

   * Store extracted reviews into **`Reviews.csv`** for further use.

---

## 🌐 Website for Fake Review Detection (Checkpoint 4 – In Progress)

The goal is to create a website that allows users to input a product URL, scrape its reviews, and classify them as **Real** or **Fake**.

### Components

1. **Frontend Development**

   * Input field for product URL.
   * Button to start scraping & classification.
   * Display results: review text + classification (Real/Fake).
   * Tech Options: React, Angular, Vue.js, or basic HTML/CSS/JS.

2. **Backend Development**

   * Use web scraping script from Checkpoint 3.
   * Load trained ML models from Checkpoint 2.
   * Create an **API endpoint**:

     * Accepts product URL.
     * Scrapes reviews.
     * Classifies them.
     * Returns JSON response.

3. **Data Flow**

   * User enters product URL → Frontend.
   * Backend scrapes reviews + classifies them.
   * Backend sends JSON → Frontend displays results.

⚠️ **Current Status**:
Checkpoint 4 is **not yet fully implemented**. Scraping and model classification work independently, but integration into a web application is pending.

---

## 🚀 Future Work

* Complete **Checkpoint 4** (Frontend + Backend integration).
* Deploy the web app using **Flask/FastAPI + React/HTML frontend**.
* Add visualization dashboards for classification statistics.
* Extend dataset with more real-world reviews for higher accuracy.

---

## 💾 Model Saving & Loading

```python
import joblib
joblib.dump(pipeline, 'fake_review_model.pkl')
```

Models can be reloaded anytime for predictions without retraining.

---

✅ **Status Summary**

* ✔️ Data preprocessing done
* ✔️ Model training & evaluation completed
* ✔️ Web scraping implemented
* ⏳ Website integration (Checkpoint 4) still in progress

---
