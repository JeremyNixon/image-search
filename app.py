from flask import Flask, render_template
import os
from random import random
import numpy as np
import fnmatch

app = Flask(__name__)

def weighted_rand(spec):
    i, sum = 0, 0
    r = random()
    for i in spec:
        sum += spec[i]
        if r <= sum:
            return int(i)

def find_images_in_folder(folder_path, image_extensions=('*.jpg', '*.jpeg', '*.png', '*.gif')):
    image_files = []
    for root, _, files in os.walk(folder_path):
        for ext in image_extensions:
            for filename in fnmatch.filter(files, ext):
                full_filepath = os.path.join(root, filename).split('static/')[1]
                image_files.append((full_filepath, filename))
    return image_files

@app.route('/')
def index():
    img_dir = '/home/ga/static/instagram_training_data/'
    # img_dir = '/home/ga/static/sd/txt2img-grids/'
    # img_dir = '/home/ga/static/sd/img2img-images/'
    # img_dir = '/home/ga/static/sd/img2img-grids/'
    img_paths = os.listdir(img_dir)[:50]
    as_np = np.array(find_images_in_folder(img_dir))
    chosen_indices = as_np[np.random.randint(0, len(as_np), 500)]
    images = list(chosen_indices)
    print(images[:5])
    # images = [
    #     'image1.jpg',
    #     'image2.jpg',
    #     'image3.jpg',
    #     'image4.jpg',
    # ]
    return render_template('index.html', images=images, weighted_rand=weighted_rand)

if __name__ == '__main__':
    app.run(debug=True)