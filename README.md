# Recomendation_systems


A simple recommendation system uses matrix-factorization to extract latent features and topk ranking based on the predicted ratings.

# Install:

create virtual enviroment

virttualenv env
source env/bin/activate

pip install .

# Usage:

from app import cli

cli.optimize()

cli.train_model_app()

cli.predict()

or

cli.recommendation(item_id=item_id, top_k=top_k)

# FastAPI:

uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload

Follow the link http://0.0.0.0:8000 then add /docs after it. Have fun to play around with fastapi-UI !!! 