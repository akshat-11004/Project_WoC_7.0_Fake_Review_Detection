# Review Data Preprocessing and Analysis

In this checkpoint my goal is to clean, normalize, and convert the data into numerical vectors for further analysis and machine learning tasks.

## Steps in Preprocessing

### 1. **Loading Data**
The dataset is sourced from our mentors and loaded into the environment for processing.

### 2. **Data Cleaning**
- **Handling Missing Values**: Missing values were removed or imputed to ensure a clean dataset.
- **Removing Duplicates**: Duplicate rows were identified and removed to avoid redundancy.
- **Eliminating Irrelevant Entries**: Irrelevant rows or columns that did not contribute to the analysis were dropped.

### 3. **Text Normalization**
- Converted all text to lowercase.
- Removed punctuation, special characters, and numbers to retain meaningful words.

### 4. **Tokenization**
- Split reviews into individual words for better processing and analysis.
- Removed stopwords to focus on significant terms.

### 5. **Stemming and Lemmatization**
- Applied stemming or lemmatization to reduce words to their base forms, ensuring uniformity in the dataset.

### 6. **Feature Representation**
The text data was converted into numerical vectors for machine learning:
- **Bag-of-Words (BoW)**: A simple method capturing word frequency.
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Weighed word importance relative to the corpus.
- **Embeddings**: Dense representations capturing semantic relationships between words using pre-trained models like GloVe.

## Dataset Export
The preprocessed dataset was saved into a CSV file for future use in future work.

## Key Learnings and Recommendations
- **Method Selection**: Word Embeddings is use for this tasks like sentiment analysis or context understanding.
- **Scalability**: Preprocessing steps like tokenization, stopword removal, and lemmatization significantly improve dataset quality and model performance.
