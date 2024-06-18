import os
import json
import requests
from tqdm import tqdm

def download_image(image_url, folder, image_id):
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Define the image's filename based on its ID
    filename = f"{image_id}.jpg"
    filepath = os.path.join(folder, filename)

    # Download and save the image
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(filepath, 'wb') as file:
            file.write(response.content)
    else:
        print(f"Error downloading {filename} from {image_url}")

def download_images_from_json(json_file_path, output_folder):
    # Load the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Iterate through each entry in the JSON data
    for item in tqdm(data):
        image_id = item.get('image_id')
        if image_id:
            # Construct the URL for the image; adjust as necessary
            image_url = f"http://images.cocodataset.org/train2014/COCO_train2014_{str(image_id).zfill(12)}.jpg"
            download_image(image_url, output_folder, image_id)

# Specify the JSON file and the output folder
json_file_path = './translated_partial.json'  # Adjust as necessary
output_folder = './downloaded_images'  # Adjust as necessary

# Download the images
download_images_from_json(json_file_path, output_folder)
