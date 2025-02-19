from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# ScraperAPI Key (Replace with your actual API key)
API_KEY = "f25c1a14bf00ccbc33850356f9f1f3a7"

# Function to get HTML content of the Amazon page
def get_amazon_reviews(product_url):
    api_url = f"http://api.scraperapi.com?api_key={API_KEY}&url={product_url}&render=true"
    response = requests.get(api_url)

    if response.status_code == 200:
        return response.text  # Return raw HTML content
    else:
        return None

# Function to extract reviews from HTML
def extract_reviews(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    reviews = soup.find_all("span", {"data-hook": "review-body"})
    return [review.text.strip() for review in reviews]

@app.route("/", methods=["GET", "POST"])
def index():
    reviews = []
    if request.method == "POST":
        product_url = request.form["product_url"]
        html_content = get_amazon_reviews(product_url)
        if html_content:
            reviews = extract_reviews(html_content)
        else:
            reviews = ["Failed to fetch reviews. Please check the product link or API key."]
    return render_template("index.html", reviews=reviews)

if __name__ == "__main__":
    app.run(debug=True)
    
