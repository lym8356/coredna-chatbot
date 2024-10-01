from llama_index.core.tools import FunctionTool, ToolMetadata
import urllib.parse
from constants import SHARPSPRING_ENDPOINT

"""
This method takes in a list of field values and an embed code, URI encode it and return an unique sharpspring URL
"""

def create_sharpspring_url(ss_details: dict):
    base_url = SHARPSPRING_ENDPOINT
    embed_code = ss_details.get("embedCode", "")

    # Remove embedCode from the dictionary for constructing the query string
    ss_fields = {k: v for k, v in ss_details.items() if k.startswith("field_")}

    ss_string = urllib.parse.urlencode(ss_fields)

    request_url = f"{SHARPSPRING_ENDPOINT}{urllib.parse.quote(embed_code)}/jsonp/?{ss_string}"

    return request_url



def sharpspring_tool():
    return FunctionTool(
        fn=create_sharpspring_url,
        metadata=ToolMetadata(
            name="sharpspring_tool",
            description=(
                "This tool creates a sharpspring URL that can be used by the http_request tool, it takes a embed code and a list fields that need to be process"
            )
        )
    )