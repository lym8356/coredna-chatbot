from llama_index.core.tools import FunctionTool, ToolMetadata
import requests

def http_request(url: str, method: str, data: dict = None):

    """
    Sends an HTTP request to the provided URL with the specified method.
    Supports GET, POST, PUT, and DELETE requests.
    The data parameter is a dictionary object used for POST and PUT requests.
    """

    method = method.upper()
    if method not in ['GET', 'POST', 'PUT', 'DELETE']:
        raise ValueError(f"Invalid HTTP method: {method}. Supported methods are GET, POST, PUT, DELETE.")

    try:
        if method == 'GET':
            response = requests.get(url)
        elif method == 'POST':
            response = requests.post(url, json=data)
        elif method == 'PUT':
            response = requests.put(url, json=data)
        elif method == 'DELETE':
            response = requests.delete(url)
    except requests.exceptions.RequestException as e:
        return f"HTTP request failed: {str(e)}"
    
    result = {
        "status_code": response.status_code,
        "content": response.content.decode('utf-8') if response.content else "No content"
    }

    return result


def http_tool():

    return FunctionTool(
        fn=http_request,
        metadata=ToolMetadata(
            name="http_tool",
            description=(
                "This tool allows sending HTTP requests (GET, POST, PUT, DELETE) to a given URL, if there is no url available, do not use this tool "
            )
        )
    )