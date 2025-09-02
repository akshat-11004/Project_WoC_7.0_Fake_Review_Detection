from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
import pickle
import warnings
warnings.filterwarnings('ignore')
import string, nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords")
nltk.download("punkt")

app = Flask(__name__)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def text_process(review):
    review = str(review) 
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    words = word_tokenize(nopunc)
    words = [word for word in words if word.lower() not in stopwords.words('english')]
    words = [stemmer.stem(word) for word in words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

with open("D:\downloads\WOC\Checkpoint_4\preprocessing_pipeline (1).pkl", "rb") as glove_file:
    preprocessing_pipeline = pickle.load(glove_file)

with open("D:\downloads\WOC\Checkpoint_4\svc_pipeline.pkl", "rb") as model_file:
    model = pickle.load(model_file)

API_KEY = "78439d08049959f43a6861987d5e18e9"

# Function to get HTML content of the Amazon page
def get_amazon_reviews(product_url):
    api_url = f"http://api.scraperapi.com?api_key={API_KEY}&url={product_url}&country_code=us&render=true"

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()  # Raise error for bad responses
        return response.text  # Return HTML
    except requests.exceptions.RequestException as e:
        print(f"Error fetching reviews: {e}")
        return None

# Function to extract reviews from HTML
def extract_reviews(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    reviews = soup.find_all("span", {"data-hook": "review-body"})
    if not reviews:
        print("No reviews found. Possible HTML structure change.")
    return [review.text.strip() for review in reviews]

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    
    if request.method == "POST":
        product_url = request.form.get("product_url", "").strip()
        
        if not product_url:
            results = [("Invalid URL. Please enter a valid product link.", "N/A")]
        else:
            html_content = get_amazon_reviews(product_url)
            if html_content:
                reviews = extract_reviews(html_content)
                
                # if not hasattr(preprocessing_pipeline, "idf_"):  # Check if already fitted
                #     print("Fitting vectorizer...")
                # preprocessing_pipeline = TfidfVectorizer()
                preprocessing_pipeline.transform(reviews)  # Train with extracted reviews

                # Transform the extracted reviews into vectors
                review_vectors = preprocessing_pipeline.transform(reviews)

                # review_vectors = preprocessing_pipeline.transform(reviews)  # ✅ Use the loaded vectorizer
                if reviews:
    
                    if not hasattr(preprocessing_pipeline, "idf_"):  # IDF check
                        print("Vectorizer is not fitted. Training now...")
                        # preprocessing_pipeline.transform(reviews)  # Train it on extracted reviews

                    # Convert extracted reviews into vectorized format
                    # review_vectors = preprocessing_pipeline.transform(reviews)

                    # Predict using the trained model
                    predictions = model.predict(review_vectors)
                    
                    results = list(zip(reviews, ["Real" if p == 0 else "Fake" for p in predictions]))
                else:
                    results = [("No reviews found.", "N/A")]
            else:
                results = [("Failed to fetch reviews. Please check the product link or API key.", "N/A")]

    return render_template("index1.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
