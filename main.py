import requests
from tqdm import tqdm

def download_file(url, filename):
    # First, send a HEAD request to get the file size
    response = requests.head(url)
    file_size = int(response.headers.get('Content-Length', 0))

    # Set up a progress bar
    progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True)

    # Now, download the file with streaming
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as file:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            progress_bar.update(len(data))
    progress_bar.close()

    if file_size != 0 and progress_bar.n != file_size:
        print("ERROR, something went wrong")

# Example URL for COCO dataset annotations (Update URL as needed)
annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
# Specify the local filename where you want to save the download
local_filename = "annotations_trainval2014.zip"

download_file(annotations_url, local_filename)
