# Amazon Product Reviews Scraping

This demonstrates how to scrape Amazon product reviews using an API-based approach.

## **Overview**

This process involves retrieving Amazon product reviews through a **ScraperAPI** (or similar third-party API) that bypasses Amazon's restrictions. The scraped reviews are stored in a text file for later analysis or use.

## **Steps to Scrape Amazon Reviews**

1. **Send a Request to ScraperAPI**
   - Send the product page URL to ScraperAPI to fetch the page content. The response will include the HTML of the Amazon page.
   - This HTML contains all product reviews, which we can then parse and extract.

2. **Extract Reviews Using BeautifulSoup**
   - Parse the HTML content using **BeautifulSoup** to find the review containers (`<span>` tags with the `data-hook="review-body"` attribute).
   - Extract the review text and clean it up (e.g., remove extra spaces).

3. **Save Reviews to a File**
   - Store the extracted reviews in a **text file** (`amazon_reviews.txt`) for easy access.
   - Each review is saved on a new line with extra space between them to maintain readability.

## **Considerations**
- **API Limitations**: Free tiers of APIs may have limitations on the number of requests.
- **Handling CAPTCHAs**: Some anti-bot measures may still apply depending on the service or frequency of requests.
- **Legal Compliance**: Always check the Terms of Service of the website or service you are scraping to ensure compliance with their policies.

## **References**
- [ScraperAPI Documentation](https://www.scraperapi.com/)
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
