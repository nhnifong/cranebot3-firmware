import json
import requests

fields = ['Content-Type', 'Content-Length', 'X-Timestamp-Sec', 'X-Timestamp-Usec']

def parse_mixed_replace_stream(url, part_cb):
    """
    Parses a multipart/x-mixed-replace stream using the requests library.

    Args:
        url: The URL of the stream.
        part_cb: function accepting a dict of headers and a bytes
            called for every part that is received from the stream

    Prints the content type of each part in the stream.
    """
    with requests.get(url, stream=True) as response:
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '')
        if "multipart/x-mixed-replace" not in content_type:
            print("Error: Not a multipart/x-mixed-replace stream.")
            return
        boundary = content_type[content_type.find('boundary=')+9:]
        boundary_with_newlines = f"\r\n{boundary}\r\n"

        for part in response.iter_lines(chunk_size=2**10, decode_unicode=False, delimiter=boundary_with_newlines.encode()):
            headers = {}
            lines = part.split(b'\r\n')
            for line in lines:
                line = line.decode()
                s = line.split(': ')
                if len(s) == 2:
                    headers[s[0]] = s[1]
                else:
                    break
            part_cb(headers, lines[-1])

# URL = "http://127.0.0.1:8001/stream"
# parse_mixed_replace_stream(URL, lambda a,b: print(a))