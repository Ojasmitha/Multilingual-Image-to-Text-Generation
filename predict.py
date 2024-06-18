import os
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from tqdm import tqdm
vector_dir = 'vectors'
models = os.listdir(vector_dir)
print(models)

vectors = {}
for model in models:
    vec_files = [file for file in os.listdir(os.path.join(vector_dir, model)) if len(file.split('_')) == 3]
    vectors[model] = {'captions': {}, 'image': None}
    langs = [file.split('_')[0] for file in vec_files]
    for lang, filename in zip(langs, vec_files):
        with open(os.path.join(vector_dir, model, filename), 'r') as f:
            vectors[model]['captions'][lang] = [np.array([float(num) for num in line.split()]) for line in f.readlines()]

    with open(os.path.join(vector_dir, model, 'image_vectors.tsv'), 'r') as f:
        vectors[model]['image'] = [np.array([float(num) for num in line.split()]) for line in f.readlines()]
en_df = pd.DataFrame({lang:vecs for lang, vecs in vectors['en-clip-model-0.01-3-0.5.pt']["captions"].items()})
en_df['img'] = vectors['en-clip-model-0.01-3-0.5.pt']['image']
print(en_df.head())
print(en_df.columns)
cols = en_df.columns.to_list()
cols = en_df.columns.to_list()
cols.remove('img')
en_k_scores = {}
for k in [1,5,10]:
    en_k_scores[k] = []
    print(k)
    for lang in cols:
        print(lang)
        en_dists = euclidean_distances(en_df[lang].to_list(), en_df['img'].to_list())
        en_acc = top_k_accuracy_score(np.arange(len(en_dists)), en_dists*(-1), k=k)
        en_k_scores[k].append(en_acc)
        print(en_acc)
        print()
with open('translated_partial.json', 'r') as f:
    captions = json.load(f)

lang = 'bn'
for model, df in [("English", en_df)]:
    print(model)
    dists = euclidean_distances(df[lang].to_list(), df['img'].to_list())
    print("Euclidean Distances Matrix:")
    print(dists)
    
    diag = np.diag(dists)
    top_5 = np.argsort(diag)[:5]
    print("Top 5 Diagonal Distances:")
    print(diag[top_5])

    back_5 = np.argsort(diag)[-5:]
    print("Bottom 5 Diagonal Distances:")
    print(diag[back_5])

    true_idx = np.arange(len(dists))

    # The rank of the true image
    ranks = np.argsort(dists, axis=1)
    true_rank = np.array([ranks[idx].tolist().index(idx) for idx in true_idx])
    indices = np.argsort(true_rank)
    top_5 = indices[:5]
    back_5 = indices[-5:]


with open('translated_partial.json', 'r') as f:
    captions = json.load(f)



lang = 'bn'
for model, df in [("English", en_df)]:
    print(model)
    dists = euclidean_distances(df[lang].to_list(), df['img'].to_list())
    diag = np.diag(dists)

    top_5 = np.argsort(diag)[:5]
    print(top_5)
    print(diag[top_5])

    back_5 = np.argsort(diag)[-5:]
    print(back_5)
    print(diag[back_5])
    dists = euclidean_distances(df[lang].to_list(), df['img'].to_list())
    true_idx = np.arange(len(dists))

    # The rank of the true image
    ranks = np.argsort(dists, axis=1)
    true_rank = np.array([ranks[idx].tolist().index(idx) for idx in true_idx])
    indices = np.argsort(true_rank)
    top_5 = indices[:5]
    back_5 = indices[-5:]

def load_image(img_id):
    for image in os.listdir('data/images/downloaded_images'):
        if image.endswith(f'{img_id}.jpg'):
            return plt.imread(os.path.join('data/images/downloaded_images', image))     
with open('translated_partial.json', 'r') as f:
    captions = json.load(f)
lang = 'bn'
for model, df in [("English", en_df)]:
    print(model)
    dists = euclidean_distances(df[lang].to_list(), df['img'].to_list())
    true_idx = np.arange(len(dists))

    # The rank of the true image
    ranks = np.argsort(dists, axis=1)
    true_rank = np.array([ranks[idx].tolist().index(idx) for idx in true_idx])
    indices = np.argsort(true_rank)
    top_5 = indices[:5]
  
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    fig.tight_layout(pad=1.0)  # Add this line to adjust the spacing between subplots
    for i in range(5):
    # Access the English caption directly
     text = captions[top_5[i]]['caption']  
     img_id = captions[top_5[i]]['image_id']
    
    # Assuming you have defined a function `load_image(image_id)` that loads the image based on its ID
    image = load_image(img_id)  # Make sure this function correctly fetches and returns an image array
    
    axs[i].imshow(image)
    axs[i].set_title(text, wrap=True)  # This will display the caption as the title of each subplot
    axs[i].axis('off')  # Hide axes for a cleaner look

    plt.savefig(f'top5_{model}_{lang}.png', dpi=300)  # Save the figure to a file
    plt.show()  #

image_info = captions[np.random.randint(len(captions))]

# Print the image ID and its associated captions
print("Image ID:", image_info['image_id'])
print( image_info['caption'])

for lang, translation in image_info['translated_captions'].items():
    print(f"{lang.capitalize()}: {translation}")

# Define the vector for the selected image
image_vector = np.random.rand(1, 768)  # Assuming a vector of shape (1, 768) for illustration

# Define the vectors for the captions
caption_vectors = [np.random.rand(1, 768) for _ in range(len(captions))]  # Replace with actual caption vectors

# Calculate the distances between the selected image and all captions
distances = euclidean_distances(image_vector, caption_vectors)

# Print the distances
print("Distances to Captions:", distances)