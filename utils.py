from bs4 import BeautifulSoup
import requests, os
from typing import List, cast, Optional
from llama_index.core import (
    Document
)
from llama_index.readers.web import SitemapReader
from llama_index.readers.file import PDFReader
from llama_index.core import SimpleDirectoryReader

from constants import SIMPLE_DIRECTORY_READER_SUPPORTED_TYPES, DATA_DIR

def check_if_css_class_exists(url: str, css_class: str):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    forms = soup.find_all('a')

    return len(forms)


def load_data_from_files(
        file_names: Optional[List[str]] = None,
        directory: Optional[str] = None,
) -> List[Document]:
    """
    Load data from a file and return a list of LlamaIndex Document objects.

    Args:
        file (str): The path to the file to be processed.

    Returns:
        List[Document]: A list of LlamaIndex Document objects.

    Raises:
        ValueError: If no file is provided or the file type is unsupported.
    """
    
    documents = []

    # If file_names are provided, process each file
    if file_names:
        if not isinstance(file_names, list) or not all(isinstance(f, str) for f in file_names):
            raise ValueError("file_names must be a list of strings.")

        for file_name in file_names:
            file_path = os.path.join(DATA_DIR, file_name)
            
            if not os.path.isfile(file_path):
                raise ValueError(f"The file '{file_path}' does not exist.")
            
            # Get the file extension and process accordingly
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension in SIMPLE_DIRECTORY_READER_SUPPORTED_TYPES:
                docs = SimpleDirectoryReader(input_dir=os.path.dirname(file_path)).load_data()
                documents.append(docs)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
    
    # If no file_names are provided, check for directory and load all files from it
    elif directory:
        directory_path = os.path.join(DATA_DIR, directory)
        if not os.path.isdir(directory_path):
            raise ValueError(f"The directory '{directory}' does not exist.")
        
        # Load all files in the directory
        docs = SimpleDirectoryReader(input_dir=directory).load_data()
        documents.append(docs)

    else:
        raise ValueError("Either file_names or directory must be provided.")

    return documents


def load_data_from_sitemap(sitemap_url: str, domain: str) -> List[Document]:

    """
    Crawl URLs in the sitemap and convert them to Document objects.
    
    Args:
        sitemap_url (str): The URL of the sitemap to crawl.
        domain (str): The domain to filter the sitemap URLs.
    
    Returns:
        List[Document]: A list of Document objects converted from the urls stored in the sitemap.
    
    Raises:
        ValueError: If the provided sitemap URL or domain is invalid.
        ConnectionError: If there's a problem connecting to the sitemap URL.
        RuntimeError: If there's an issue during the loading or processing of the sitemap data.
    """

    if not isinstance(sitemap_url, str) or not sitemap_url.strip():
        raise ValueError("Invalid URL: The provided URL is either None, not a string, or empty.")
    
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError("Invalid domain: The provided domain is either None, not a string, or empty.")
    
    # continue processing sitemap urls
    try:
        response = requests.head(sitemap_url)
        if response.status_code != 200:
            raise ConnectionError(f"Failed to access the sitemap URL: {sitemap_url}. Status code: {response.status_code}")
        
        loader = SitemapReader()

        documents = loader.load_data(
            sitemap_url=sitemap_url,
            filter=domain
        )

        return documents
    
    except requests.RequestException as error:
        raise ConnectionError(f"Error occurred while trying to access the URL provided: {error}")
    
    except BaseException as error:
        print(f"An error occurred while processing the sitemap: {str(error)}")
    