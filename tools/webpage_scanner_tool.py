import requests
from llama_index.core.tools import FunctionTool, ToolMetadata
from bs4 import BeautifulSoup


def fetch_from_input_name(input_name: str, url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all input elements with names starting with the specified pattern
        inputs = soup.find_all('input', attrs={'name': lambda x: x and x.startswith(input_name)})
        
        result = []

        for input_elem in inputs:
            field_name = input_elem.get('name', '')
            placeholder_name = input_elem.get('placeholder', '')
            result.append({field_name: placeholder_name})

        return inputs

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return []

def check_class_exists(class_name: str, url: str):
    pass

def check_tag_exists(tag_name:str, url:str):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    elements = soup.find_all(tag_name)

    return len(elements) > 0


def fetch_field_tool():
    return FunctionTool(
        fn=fetch_from_input_name,
        metadata=ToolMetadata(
            name="fetch_field_tool",
            description=(
                "This tool fetches HTML input elements from a URL where the 'name' attribute starts with a given input name. It returns a list of dictionaries, each containing the 'name' and 'placeholder' attributes of the input elements. If an error occurs, it returns an empty list."
            )
        )
    )

def class_exists_tool():
    return FunctionTool(
        fn=check_class_exists,
        metadata=ToolMetadata(
            name="class_exists_tool",
            description=(
                "This tool checks if a specific css class exists in a url, it returns true of false, pass in a tag_name and a url as parameters"
            )
        )
    )

def tag_exists_tool():
    return FunctionTool(
        fn=check_tag_exists,
        metadata=ToolMetadata(
            name="tag_exists_tool",
            description=(
                "This tool checks if a specific html tag exists in a url, it returns true of false"
            )
        )
    )