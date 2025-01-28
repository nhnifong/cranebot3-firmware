import json
import requests

def parse_mixed_replace_stream(url):
    """
    Parses a multipart/x-mixed-replace stream using the requests library.

    Args:
        url: The URL of the stream.

    Prints the content type of each part in the stream.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        boundary = b"\r\n--123456789000000000000987654321\r\n"

        if "multipart/x-mixed-replace" not in response.headers.get('Content-Type', ''):
            print("Error: Not a multipart/x-mixed-replace stream.")
            return

        for chunk in response.iter_content(chunk_size=2**12):
            if boundary in chunk:
                try:
                    start_index = chunk.index(boundary) + len(boundary)
                    end_index = chunk.index(b"\r\n\r\n", start_index)
                    headers = chunk[start_index:end_index].decode('utf-8')

                    # Extract Content-Type from headers
                    content_type = None
                    for line in headers.splitlines():
                        if line.lower().startswith('content-type:'):
                            content_type = line.split(':')[1].strip()
                            break

                    if content_type:
                        print(f"Content-Type: {content_type}")
                except ValueError:
                    # Handle cases where boundary is not found or headers are not properly formatted
                    pass

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

URL = "http://127.0.0.1:8001/stream"
parse_mixed_replace_stream(URL) 