from observer import parse_mixed_replace_stream
import sys

print(f"http://%s/stream" % sys.argv[1])
parse_mixed_replace_stream(f"http://%s/stream" % sys.argv[1])

#192.168.1.148:80