# Datasets
ML_SLUG=prajitdatta/movielens-100k-dataset
ML_SUBDIR=ml-100k
TMDB5000_SLUG=tmdb/tmdb-movie-metadata


# Embeddings
EMB_MODEL=sentence-transformers/all-MiniLM-L6-v2

# LLM (Ollama)
OLLAMA_URL=http://host.docker.internal:11434
OLLAMA_MODEL=qwen2.5:14b-instruct

# UI -> API
UI_API_BASE=http://api:8000

# ---- API / UI ports (container → host mapping controlled in compose) ----
API_HOST=0.0.0.0
API_PORT=8000
UI_PORT=8501

# keep GPU off in this slim image
CUDA_VISIBLE_DEVICES=

# Keys
OMDB_API_KEY=""                #http://www.omdbapi.com/apikey.aspx
TMDB_API_KEY=""