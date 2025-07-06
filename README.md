#Consumer Complaint Categorization using NLP
This project is about analyzing and categorizing real-world consumer complaints using Natural Language Processing (NLP). The goal was to classify each complaint into its corresponding product category based on the text of the complaint. It's built using Python and popular libraries like Pandas, Scikit-learn, and Matplotlib.

About the Dataset
The data comes from the Consumer Financial Protection Bureau (CFPB) and contains over 1.4 million consumer complaints related to financial products. For quicker processing and training, I used a sample of 10,000 records for this project.

Each record in the dataset includes:

The product category (like Credit card, Mortgage, Student loan, etc.)

A written narrative describing the consumer's complaint

What This Project Covers
1. Data Cleaning
Selected only relevant columns: product type and complaint narrative

Removed missing values and unnecessary characters

Converted text to lowercase and removed common stopwords (like “the”, “is”, “and”)

2. Text Vectorization
Used TF-IDF (Term Frequency–Inverse Document Frequency) to convert the cleaned text into numerical features

Limited to the top 5000 most frequent words to keep things efficient

3. Label Encoding
The product categories were converted to numeric labels so they could be used in training

4. Model Training
Used Logistic Regression for classification

The model was trained on 80% of the data and tested on the remaining 20%

5. Evaluation
Evaluated performance using a classification report (precision, recall, F1-score)

Also plotted a confusion matrix to visualize how well the model is performing across categories

6. Saving the Model
Exported the trained model and vectorizer using joblib so they can be reused later without retraining

Files in This Repository
File	Description
consumer_complaint_classifier.ipynb	The complete training notebook with all steps
logistic_model.pkl	The saved logistic regression model
tfidf_vectorizer.pkl	The saved TF-IDF vectorizer used to process text

How to Use This Model
Once everything is loaded, you can use the following function to predict the category of any new complaint text:

python
Copy code
import joblib

model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def clean_text(text):
    # Same cleaning logic used during training
    import re
    stop_words = set("a an the and is was were to from with for of".split())
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def predict_complaint_category(text):
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)
    return prediction[0]
