# env.sh
export DB_PATH_V1="/mnt/a/movies/movies_v1/datawrangling/movies.db"  
export DB_PATH_V2="/mnt/a/movies/movies_v2/datawrangling/movies.db"  
export DATASET_SLUG="prajitdatta/movielens-100k-dataset"
export DATA_SUBDIR="ml-100k"
export OMDB_API_KEY="<your_omdb_key>" #http://www.omdbapi.com/apikey.aspx
export TMDB_API_KEY="<your_tmdb_key>" #https://www.themoviedb.org/settings/api
export EMB_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export FAISS_PATH_V1="/mnt/a/movies/movies_v1/datawrangling/index/movies.faiss"
export FAISS_PATH_V2="/mnt/a/movies/movies_v2/datawrangling/index/movies.faiss"
export OLLAMA_URL="http://127.0.0.1:11434"
export LLM_MODEL_NAME="qwen2.5:14b-instruct"
export API_BASE="http://127.0.0.1:8000"