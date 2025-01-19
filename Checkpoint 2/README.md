# Fake Review Detection

This project aims to build a model to detect fake reviews using machine learning. The primary approach involves text preprocessing, embedding generation, and classification using various machine learning models such as Random Forest, SVM, and Logistic Regression. The pipeline ensures proper handling of text data and delivers predictions on whether a review is genuine or fake.

## Training the Model

The training process involves splitting the dataset into training and testing sets, followed by building machine learning pipelines for each model. These pipelines ensure that the preprocessing steps are applied before the model is trained. The models used in this project are:

1. **Random Forest Classifier**: An ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting.
2. **Support Vector Machine (SVM)**: A powerful classification algorithm that finds the optimal hyperplane separating the classes.
3. **Logistic Regression**: A linear model used for binary classification tasks, suitable for predicting probabilities.

### Steps for Training the Model:

1. **Data Splitting**:
   - The dataset is divided into two sets: one for training the models and the other for testing. This ensures that the models are trained on one portion of the data and evaluated on another, preventing overfitting.

    ```python
    from sklearn.model_selection import train_test_split

    X = df['embedding'].tolist()  # Features: sentence embeddings
    y = df['label'].apply(lambda x: 1 if x == 'CG' else 0)  # Target: 1 for 'CG', 0 for 'OR'
    
    # Split data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

2. **Pipeline Creation**:
   - A pipeline is defined for each model, which combines both the feature scaling (StandardScaler) and the classifier (Random Forest, SVM, or Logistic Regression). This ensures that the same preprocessing steps are applied during both training and prediction.

    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Define a pipeline for each model
    pipelines = {
        "Random Forest": Pipeline([
            ('scaler', StandardScaler()),  # Standardize features
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        "SVM": Pipeline([
            ('scaler', StandardScaler()),  # Standardize features
            ('classifier', SVC(kernel='linear', random_state=42))
        ]),
        "Logistic Regression": Pipeline([
            ('scaler', StandardScaler()),  # Standardize features
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
    }
    ```

3. **Model Training and Evaluation**:
   - The models are trained on the training data, and their performance is evaluated on the test data. Evaluation is performed using a **classification report**, which includes metrics such as accuracy, precision, recall, and F1-score. These metrics provide a comprehensive understanding of how well the model is performing.

    ```python
    from sklearn.metrics import classification_report

    # Train and evaluate each model
    for model_name, pipeline in pipelines.items():
        print(f"\nTraining and evaluating {model_name}...")
        # Fit the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Evaluate the model
        print(f"Classification Report for {model_name}:\n")
        print(classification_report(y_test, y_pred))
    ```

4. **Hyperparameter Tuning (Optional)**:
   - To improve the performance of the models, hyperparameter tuning can be performed using techniques like **GridSearchCV**. This step searches for the best combination of hyperparameters for each model, potentially improving accuracy.

    ```python
    from sklearn.model_selection import GridSearchCV

    # Example for Random Forest model
    param_grid = {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [10, 20, None]}
    grid_search = GridSearchCV(pipelines['Random Forest'], param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for Random Forest: {grid_search.best_params_}")
    ```

5. **Model Evaluation**:
   - After training the models, their performance is evaluated based on the **test data**. The classification report provides important metrics like:
     - **Precision**: The proportion of true positive results among all positive predictions.
     - **Recall**: The proportion of true positive results among all actual positive instances.
     - **F1-Score**: The harmonic mean of precision and recall.
     - **Accuracy**: The overall correctness of the model.

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

## Saving and Loading Models

After training the models, they are saved using `joblib`. This ensures that the models can be loaded and used for prediction without the need for retraining.

