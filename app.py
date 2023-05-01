from flask import Flask, render_template, request
import os
from random import random
import numpy as np
import fnmatch
import os
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
<<<<<<< HEAD
=======
import time
from transformers import CLIPProcessor, CLIPModel
>>>>>>> bb3cc5d (make code run on one machine)

app = Flask(__name__)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def weighted_rand(spec):
    i, sum = 0, 0
    r = random()
    for i in spec:
        sum += spec[i]
        if r <= sum:
            return int(i)

def load_data(file_path):
    with open(file_path, 'rb') as f:
        index, np_embeddings, path_to_index, index_to_path = pickle.load(f)
    return index, np_embeddings, path_to_index, index_to_path
faiss_index, np_embeddings, path_to_index, index_to_path = load_data('data.pkl')

def find_images_in_folder(folder_path, image_extensions=('*.jpg', '*.jpeg', '*.png', '*.gif')):
    image_files = []
    for root, _, files in os.walk(folder_path):
        for ext in image_extensions:
            for filename in fnmatch.filter(files, ext):
                full_filepath = os.path.join(root, filename).split('static/')[1]
                image_files.append((full_filepath, filename))
    return image_files

# Define the search function to find the k nearest neighbors
def search(embedding, k=5):
    # Ensure the input embedding has the correct data type
    embedding = np.array(embedding).astype('float32').reshape(1, -1)
    # Perform the search
    distances, indexes = faiss_index.search(embedding, k)
    return distances, indexes

def find_nearest_paths(input_path, k=100):    
    # Get the embedding for the input path
    input_path = '/home/ga/static/' + input_path
    input_embedding = np_embeddings[path_to_index[input_path]]
    # Perform the search
    distances, indexes = search(input_embedding, k)
    # Convert the indexes to paths
    nearest_paths = [index_to_path[idx] for idx in indexes.flatten()]
    return [p.split('static/')[1] for p in nearest_paths]

<<<<<<< HEAD
=======
def embed_text(text):
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
            embedding = clip_model.get_text_features(**inputs).numpy().tolist()
    return embedding

@app.route('/search', methods=['GET'])
def query_search():
    start = time.time()
    search_query = request.args.get('query')
    print("Search query received:", search_query, time.time()-start, flush=True)
    query_embedding = embed_text(search_query)
    distances, indexes = ann_search(query_embedding, 50)
    # Convert the indexes to paths
    nearest_paths = [index_to_path[idx] for idx in indexes.flatten()]
    similar_image_paths = [p.split('static/')[1] for p in nearest_paths]    
    print("Similar paths found", time.time()-start, flush=True)
    return render_template('similar_images.html', images=similar_image_paths, weighted_rand=weighted_rand)


>>>>>>> bb3cc5d (make code run on one machine)
@app.route('/similar-images')
def similar_images():
    image_path = request.args.get('image_path')
    print("Backend image path recieved:", image_path, flush=True)
    similar_image_paths = find_nearest_paths(image_path)
    return render_template('similar_images.html', images=similar_image_paths, weighted_rand=weighted_rand)


@app.route('/nearest_images')
def nearest_images():
    input_path = request.args.get('path')
    img_paths = find_nearest_paths(input_path)
    return render_template('index.html', images=img_paths)

@app.route('/')
def index():
    img_dir = '/home/ga/static/instagram_training_data/'
    img_paths = os.listdir(img_dir)[:300]
    as_np = np.array(find_images_in_folder(img_dir))
    chosen_indices = as_np[np.random.randint(0, len(as_np), 500)]
    images = list(chosen_indices)
    print("Images", images[:5])
    nearest_paths = find_nearest_paths(images[0][0])
    print("Nearest paths", nearest_paths)
    return render_template('index.html', images=nearest_paths, weighted_rand=weighted_rand)

if __name__ == '__main__':
    app.run(debug=True)