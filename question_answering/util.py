import re

def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9]', '_', filename)