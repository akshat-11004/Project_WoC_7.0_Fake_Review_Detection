import requests
import json

# Replace with your ScraperAPI key
API_KEY = "605768c3b86850469d4b6af79a974f4b"

# Function to get reviews
def get_amazon_reviews(product_url):
    # Use ScraperAPI to bypass Amazon's restrictions
    api_url = f"http://api.scraperapi.com?api_key={API_KEY}&url={product_url}&render=true"

    # Send request to ScraperAPI
    response = requests.get(api_url)

    if response.status_code == 200:
        data = response.text  # Raw HTML of the Amazon page
        return data  # You need to parse this using BeautifulSoup or regex
    else:
        print("Failed to fetch data. Status Code:", response.status_code)
        return None

# Example: Provide an Amazon product URL
amazon_product_url = "https://www.amazon.com/Bluetooth-Headphones-KVIDIO-Microphone-Lightweight/dp/B09BF64J55/ref=sr_1_1_sspa?dib=eyJ2IjoiMSJ9.7N_ryurV910mDXYaNTwwfxh8q01LWoO5jri1Hiy8-aTwsmRNRzD-5qi92rALdPOUtar_G3MAkCDPpl-XeXXy5GTj2bnPvCD7hYj7v7oJ-yYE5Z1WnbwvYiT_vvJ46pzkG8v_6PIDCXAPXGINLY690kpwxf4KfhjcV3PEa6SA3oBQteJXF0xV9yVQ1kcarOHTIKQjmuIt8qV-I65hh_-tUPwHLp4_peSnbccqBWysmGA.OInaqPeGQ0K2_g59O1ZakOulouhb8V_DtexOZEWSQW4&dib_tag=se&keywords=head%2Bphones&qid=1738303071&sr=8-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1"  # Replace with your product link

# Get reviews
html_content = get_amazon_reviews(amazon_product_url)

# Save the HTML (optional for later parsing)
with open("amazon_reviews.html", "w", encoding="utf-8") as file:
    file.write(html_content)

print("Amazon reviews page scraped successfully. Check amazon_reviews.html")

from bs4 import BeautifulSoup

def extract_reviews(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    # Find all review containers
    reviews = soup.find_all("span", {"data-hook": "review-body"})

    extracted_reviews = [review.text.strip() for review in reviews]

    return extracted_reviews

# Extract reviews
reviews = extract_reviews(html_content)

# Save reviews to a text file
with open("amazon_reviews.txt", "w", encoding="utf-8") as file:
    for review in reviews:
        file.write(review + "\n\n")  # Adding space between reviews

print("Amazon reviews saved successfully in amazon_reviews.txt")