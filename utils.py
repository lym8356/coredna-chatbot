import requests, os
from typing import List, cast, Optional
from llama_index.core import (
    Document
)
from llama_index.readers.web import SitemapReader
from llama_index.core import SimpleDirectoryReader

from constants import SIMPLE_DIRECTORY_READER_SUPPORTED_TYPES, DATA_DIR


def load_data_from_files(
        file_names: Optional[List[str]] = None,
        directory: Optional[str] = None,
) -> List[Document]:
    """
    Load data from specified files in a directory or from all files in a directory and return a list of LlamaIndex Document objects.

    Args:
        file_names (Optional[List[str]]): List of file names to process in the directory.
        directory (Optional[str]): The directory containing the files to process.

    Returns:
        List[Document]: A list of LlamaIndex Document objects.

    Raises:
        ValueError: If only file_names are provided or if the file type is unsupported.
    """
    
    documents = []

    # Both directory and file_names are provided
    if directory and file_names:
        directory_path = os.path.join(DATA_DIR, directory)

        if not os.path.isdir(directory_path):
            raise ValueError(f"The directory '{directory_path}' does not exist.")
        
        # Process each specified file within the given directory
        for file_name in file_names:
            file_path = os.path.join(directory_path, file_name)
            
            if not os.path.isfile(file_path):
                raise ValueError(f"The file '{file_path}' does not exist.")
            
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension in SIMPLE_DIRECTORY_READER_SUPPORTED_TYPES:
                docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
                documents.extend(docs)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

    # Only directory is provided
    elif directory:
        directory_path = os.path.join(DATA_DIR, directory)

        if not os.path.isdir(directory_path):
            raise ValueError(f"The directory '{directory_path}' does not exist.")
        
        # Load all files in the specified directory
        docs = SimpleDirectoryReader(input_dir=directory_path).load_data()
        documents.extend(docs)

    # Only file_names are provided without directory
    elif file_names:
        raise ValueError("Directory must be provided when specifying file_names.")

    # Neither directory nor file_names are provided
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
    

# TODO: enforce params type
def insert_into_index(index, doc, doc_id = None):
    """
    Takes an existing Vector index, process the provided file and update that index
    
    Args:
        index: VectorIndex object, the index that needs to be updated
        doc: The document name that needs to be processed
        doc_id: The id of the document, usually is the file name
    
    Returns:
        The updated index object
    
     Raises:
        ValueError: If the document cannot be processed.
        FileNotFoundError: If the document file does not exist.
        RuntimeError: If there is an error during document insertion into the index.
    """

    # Extract directory and file name from the provided doc path
    directory = os.path.dirname(doc)
    file_name = os.path.basename(doc)

    try:
        documents = load_data_from_files(file_names=[file_name], directory=directory)
        if not documents:
            raise ValueError(f"No documents were created from the file: {doc}")
        
        for document in documents:
            if doc_id is not None:
                document.doc_id = doc_id

            index.insert(document)

        index.storage_context.persist()

        return index

    except ValueError as ve:
        # Handle specific errors related to file type and document reading
        raise ve
    except FileNotFoundError as fnf_error:
        # Handle file not found errors
        raise fnf_error
    except Exception as error:
        # Handle any other unexpected errors
        raise RuntimeError(f"An error occurred while inserting the document into the index: {error}")



def ensure_directory_exists(path):
    """Ensure the given directory path exists, create if not."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")