# Fake Review Detection Project

This project aims to detect fake reviews using Machine Learning techniques. The workflow includes data preprocessing, feature engineering, model training, and evaluation.

## Data Preprocessing

### Steps Involved:

1. **Loading Data**:
   The dataset is loaded into the environment for further processing and analysis.

2. **Data Cleaning**:
   - **Handling Missing Values**: Missing values were either removed or imputed to maintain a clean dataset.
   - **Removing Duplicates**: Duplicate entries were detected and removed.
   - **Eliminating Irrelevant Entries**: Non-contributory rows or columns were dropped to focus on essential information.

3. **Text Normalization**:
   - All text data was converted to lowercase.
   - Special characters, numbers, and punctuation were removed to ensure only meaningful words remain.

4. **Tokenization**:
   - The text data was split into individual words to facilitate analysis.
   - Stopwords were removed to retain only significant terms.

5. **Stemming and Lemmatization**:
   - Words were reduced to their base forms through stemming or lemmatization, ensuring consistency across the dataset.

6. **Feature Representation**:
   - **Bag-of-Words (BoW)**: Captures the frequency of words.
   - **TF-IDF (Term Frequency-Inverse Document Frequency)**: Weighs word importance relative to the corpus.
   - **Embeddings**: Pre-trained word embeddings (e.g., GloVe) were used to capture semantic relationships between words.

### Exporting Preprocessed Data:
The preprocessed dataset is saved into a CSV file for future analysis and model training.

### Key Learnings and Recommendations:
- **Method Selection**: Word Embeddings are ideal for tasks requiring sentiment analysis or understanding the context of the reviews.
- **Scalability**: Preprocessing steps like tokenization, stopword removal, and lemmatization significantly improve model performance by cleaning and normalizing the data.

---

## Fake Review Detection

This section describes the model and pipeline used to detect fake reviews in the dataset.

### Machine Learning Models:
1. **Random Forest Classifier**: Combines multiple decision trees to improve accuracy and reduce overfitting.
2. **Support Vector Machine (SVM)**: Finds an optimal hyperplane to separate classes.
3. **Logistic Regression**: A linear classifier for binary classification tasks, useful for predicting probabilities.

### Steps Involved in Training:

1. **Data Splitting**:
   The dataset is split into training and testing sets to prevent overfitting. 80% is used for training, and 20% for testing.

    ```python
    from sklearn.model_selection import train_test_split
    X = df['embedding'].tolist()  # Features: sentence embeddings
    y = df['label'].apply(lambda x: 1 if x == 'CG' else 0)  # Target: 1 for 'CG', 0 for 'OR'
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

2. **Pipeline Creation**:
   Pipelines are created for each model, including both feature scaling and classifier steps. The preprocessing steps are automatically applied during training and prediction.

    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipelines = {
        "Random Forest": Pipeline([
            ('scaler', StandardScaler()), 
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        "SVM": Pipeline([
            ('scaler', StandardScaler()), 
            ('classifier', SVC(kernel='linear', random_state=42))
        ]),
        "Logistic Regression": Pipeline([
            ('scaler', StandardScaler()), 
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
    }
    ```

3. **Model Training and Evaluation**:
   Each model is trained using the training set, and the performance is evaluated using a classification report, which includes metrics like accuracy, precision, recall, and F1-score.

    ```python
    from sklearn.metrics import classification_report

    for model_name, pipeline in pipelines.items():
        print(f"\nTraining and evaluating {model_name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(f"Classification Report for {model_name}:\n")
        print(classification_report(y_test, y_pred))
    ```

4. **Hyperparameter Tuning (Optional)**:
   You can perform hyperparameter tuning using GridSearchCV to find the best model parameters, which can enhance performance.

    ```python
    from sklearn.model_selection import GridSearchCV
    param_grid = {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [10, 20, None]}
    grid_search = GridSearchCV(pipelines['Random Forest'], param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for Random Forest: {grid_search.best_params_}")
    ```

5. **Model Evaluation**:
   The models are evaluated based on the classification report which contains metrics like:

    - **Precision**: Proportion of true positives among all positive predictions.
    - **Recall**: Proportion of true positives among all actual positives.
    - **F1-Score**: Harmonic mean of precision and recall.
    - **Accuracy**: Overall correctness of the model.

    **Example Output:**
    ```plaintext
Training and evaluating Random Forest...
Classification Report for Random Forest:
              precision    recall  f1-score   support

           0       0.73      0.80      0.76      4055
           1       0.77      0.70      0.73      4029

    accuracy                           0.75      8084
   macro avg       0.75      0.75      0.75      8084
weighted avg       0.75      0.75      0.75      8084


Training and evaluating SVM...
Classification Report for SVM:
              precision    recall  f1-score   support

           0       0.73      0.74      0.73      4055
           1       0.73      0.72      0.73      4029

    accuracy                           0.73      8084
   macro avg       0.73      0.73      0.73      8084
weighted avg       0.73      0.73      0.73      8084


Training and evaluating Logistic Regression...
Classification Report for Logistic Regression:
              precision    recall  f1-score   support

           0       0.73      0.73      0.73      4055
           1       0.73      0.72      0.72      4029

    accuracy                           0.73      8084
   macro avg       0.73      0.73      0.73      8084
weighted avg       0.73      0.73      0.73      8084

```

### Saving and Loading Models:
After training the models, they are saved using `joblib`, enabling future predictions without retraining.

```python
import joblib
joblib.dump(pipeline, 'fake_review_model.pkl')
