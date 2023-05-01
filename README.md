# Image Search Engine

This is an implementation of a simple, lightweight neural dense vector search powered semantic image search engine.

The model uses CLIP embeddings and FAISS to take textual search queries and to navigate through similar images by clicking on them in a simple same-energy inspired interface.

## Usage:
Add the images you'd like to index to the static/ folder.
Run the indexer.

```python3 processdata.py```

When indexing is complete, start the server:
```gunicorn3 app:app```

The webapp will be up and running at either localhost or at your server's IP address.

## Installation:

```git clone git@github.com:JeremyNixon/semantic-image-search.git```