import requests
from bs4 import BeautifulSoup
import html2text

def fetch_and_convert_to_markdown(url: str) -> str:
    try:
        # Make a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        h = html2text.HTML2Text()
        # Parse the HTML content
        html_content = response.text

        # Convert HTML to Markdown
        markdown_content = h.handle(html_content)

        return markdown_content
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching the URL: {e}")
        return ""
    except Exception as e:
        logging.error(f"Error converting HTML to Markdown: {e}")
        return ""

if __name__ == "__main__":
    # Example URL
    url = "https://www.instagram.com/p/DE-SiFQiYYv/__a%3D1"
    markdown = fetch_and_convert_to_markdown(url)
    if markdown:
        print(markdown)
    else:
        print("Failed to fetch or convert the content.")