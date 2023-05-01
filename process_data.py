import fnmatch
import os
from random import random
import numpy as np
import pandas as pd
import os
from PIL import Image
import time
import torch
import requests
import base64
import io
import faiss
import pickle
import io
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel

def find_images_in_folder(folder_path, image_extensions=('*.jpg', '*.jpeg', '*.png', '*.gif')):
    image_files = []
    for root, _, files in os.walk(folder_path):
        for ext in image_extensions:
            for filename in fnmatch.filter(files, ext):
                full_filepath = os.path.join(root, filename)
                image_files.append((full_filepath, filename))
    return image_files
paths = find_images_in_folder('/home/ga/static/')

def embed_image(image_path_or_paths):
    if isinstance(image_path_or_paths, str):
        image_paths = [image_path_or_paths]
    else:
        image_paths = image_path_or_paths

    pil_images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    inputs = processor(images=pil_images, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    return embeddings.numpy().tolist()

full_paths = [p[0] for p in paths]

processed_paths = []
embeddings = []
def write_data():
    np_embeddings = np.array(embeddings)
    print(np_embeddings.shape)
    # Assuming you already have your embeddings as np_embeddings
    np_embeddings = np.array(embeddings).astype('float32')  # Faiss requires float32 data type
    print(np_embeddings.shape)
    # (169, 512)

    # Create a flat index using the L2 distance metric (Euclidean distance)
    index = faiss.IndexFlatL2(np_embeddings.shape[1])
    # Add the embeddings to the index
    index.add(np_embeddings)

    path_to_index = {}
    index_to_path = {}
    for i, p in enumerate(processed_paths):
        path_to_index[p] = i
        index_to_path[i] = p
        
    # Save the data to disk
    with open('data2.pkl', 'wb') as f:
        pickle.dump((index, np_embeddings, path_to_index, index_to_path), f)

i = 0
for p in range(int(len(full_paths)/10)):
    if i % 100 == 0:
        print(p)
    try:
        embeddings += embed_image(full_paths[i:i+10])
        processed_paths += full_paths[i:i+10]
    except:
        pass
    if i % 1000 == 0:
        write_data()
    i += 10
    