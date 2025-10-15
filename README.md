# Movies DB query
Docker version:

add the API keys to /docker/.env.docker.sh

`docker compose build` (might take a while)

`docker compose up`


First create your virtual env (optional but recommended):

python -m venv .venv
source .venv/bin/activate

# V1
source ../../env.sh
pip install -r requirements_v1.txt

under /movies_v1
python 1.insert_movielens_into_sqlite.py
python 2.enrich_db_with_omdb.py
python 3.add_embeddings.py

## start the REST API endpoint
uvicorn main:app --reload

## run the streamlit ui
streamlit run app.py


# V2
source ../../env.sh
pip install -r requirements_v2.txt

## Creating, populating, enriching and vectorizing the DB
under /movies_v2
python 1.insert_movielens_into_sqlite.py
python 2.enrich_db_with_tmdb.py
python 3.enrich_db_with_omdb.py
python 4.add_embeddings.py

## start the REST API endpoint
uvicorn main:app --reload

## run the streamlit ui
streamlit run app.py
