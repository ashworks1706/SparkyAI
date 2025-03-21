from langchain.document_loaders import UnstructuredURLLoader
# from langchain_community.document_loaders import WebBaseLoader

def url_to_text(url: str) -> str:
    """
    Converts the content of a document from a given URL to text.

    Args:
        url (str): The URL of the document.

    Returns:
        str: The extracted text content of the document.
    """
    loader = UnstructuredURLLoader(urls=[url])
    documents = loader.load()
    return documents[0].page_content if documents else ""

# Example usage
if __name__ == "__main__":
    url = "https://asu.campuslabs.com/engage/event/11057454"
    text = url_to_text(url)
    print(text)