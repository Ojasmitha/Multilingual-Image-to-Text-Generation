import ijson
import googletrans
from tqdm import tqdm
import time

json_file_path = 'annotations/captions_train2014.json'
output_json_file_path = './translated_partial.json'

trans = googletrans.Translator()

new_data = []

with open(json_file_path, 'rb') as file:
    annotations = ijson.items(file, 'annotations.item')

    processed_count = 0
    total_translation_time = 0  # Initialize total translation time

    for annotation in tqdm(annotations):
        if processed_count >= 100:
            break

        if 'caption' in annotation:
            english_caption = annotation['caption']
            
            start_time = time.time()  # Start timing

            # Translation block
            while True:
                try:
                    fr_text = trans.translate(english_caption, dest='fr').text
                    de_text = trans.translate(english_caption, dest='de').text
                    zh_text = trans.translate(english_caption, dest='zh-cn').text
                    bn_text = trans.translate(english_caption, dest='bn').text
                    break
                except Exception as e:
                    print(f"Error during translation: {e}")
                    time.sleep(5)
            
            end_time = time.time()  # End timing
            translation_time = end_time - start_time  # Calculate duration
            total_translation_time += translation_time  # Accumulate total translation time

            annotation['translated_captions'] = {
                'fr': fr_text,
                'de': de_text,
                'zh': zh_text,
                'bn': bn_text
            }

            new_data.append(annotation)
            processed_count += 1

# Output the total translation time
print(f"Total translation time: {total_translation_time} seconds")

# Write to a new JSON file
import json  # Ensure json is imported at the beginning of your script

# Output the total translation time
print(f"Total translation time: {total_translation_time} seconds")

# Corrected: Use json.dump to write to a new JSON file
with open(output_json_file_path, 'w') as json_file:
    json.dump(new_data, json_file, indent=4)

