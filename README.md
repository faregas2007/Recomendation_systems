# Recomendation_systems
A simple recommendation system uses matrix-factorization to extract latent features and topk ranking based on the predicted ratings.
# Install:

create virtual enviroment
pip install --user .

# Usage:

from app import cli

cli.optimize()
cli.train_model_app()
cli.predict()
or
recomended = cli.recommendation(item_id=item_id, top_k=top_k)