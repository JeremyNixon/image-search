# Image Search Engine

This is an implementation of a simple, lightweight neural dense vector search powered semantic image search engine.

The model uses CLIP embeddings and FAISS to take textual search queries and to navigate through similar images by clicking on them in a simple same-energy inspired interface.


## Installation:
```
git clone https://github.com/JeremyNixon/image-search.git
cd semantic-image-search
pip install -r requirements.txt
```

## Usage:
Add the images you'd like to index to the static/ folder.
Run the indexer.

```python3 process_data.py```

When indexing is complete, start the server:

```gunicorn app:app```

The webapp will be up and running at either localhost or at your server's IP address.
