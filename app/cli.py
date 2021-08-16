import typer
import torch
import pandas as pd

import mlflow
import optuna
from optuna.integration.mlflow import MlflowCallback

from numpyencoder import NumpyEncoder

from recsys import config, eval, main, utils

warnings.filtterwarnings("ignore")

# Typer cli app
app = typer.Typer()


@app.command
def optimize(
    params_fp: Path = Path(config.config_dir, "params.json"),
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    study_name: Optional[str]= 'Optimization',
    num_trials: int=10,
)->None:
    params = Namespace(**load_dict(params_fp))

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name=study_name, direction='maximize', pruner=pruner)
    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")


    study.optimize(
        lambda trial: objective(params_fp=params_fp, device=device, trial=trial),
        n_trials=params.num_trials, 
        callbacks=[mlflow_callback],
    )

    # all trials
    print("Best value (f1):%s"%{study.best_trial.value})
    params = {**params.__dict__, **study.best_trial.params}
    #params['threshold'] = study.best_trial.user_attrs['threshold']
    save_dict(params, params_fp, cls=NumpyEncoder)
    json.dumps(params, indent=2, cls=NumpyEncoder)


@app.command
def train_model_app(
    params_fp: Path = Path(config.config_dir, "params.json"),
    model_dir: Path = Path(config.config_dir, "model"),
    experiment_name: Optional[str] = "best",
    run_name: Optional[str] = "model"
    )->None:
    """Train a model using the specified parameters

    Args:
        params_fp (Path, optional): Parameters to use for training
        model_dir (Path): location of model artifacts
        experiment_name(str, optional): Name of the experiment to save to run to.
        run_name (str, optional): Name of the run.
    """

    params = Namespace(**load_dict(params_fp))

    # start run
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id

        # train
        # notice that there is no trail. 
        artifacts = train_model(params_fp=params_fp, device=device)

        # Log metrics
        performance = artifacts['performance']
        json.dumps(performance['overall'], indent=2)
        metrics = {
            "precision": performance['overall']['precision'],
            "recall": performance['overall']['recall'],
            "f1": performance['overall']['f1'],
            "best_val_loss": artifacts['loss'],
            "HR": performance['overall']['HR'],
            "NDCG": performance['overall']['NDCG']
        }

        mlflow.log_metrics(metrics)

        # log artifacts
        with tempfile.TemporaryDirectory() as dp:
            save_dict(vars(artifacts['params']), Path(dp, "params.json"), cls=NumpyEncoder)
            save_dict(performance, Path(dp, "performance.json"))
            torch.save(artifacts['model'].state_dict(), Path(dp, "model.pt"))
            mlflow.log_artifacts(dp)
        mlflow.log_params(vars(artifacts['params']))

        open(Path(model_dir, "run_id.txt"), 'w').write(run_id)
        save_dict(vars(params), Path(model_dir, "params.json"), cls=NumpyEncoder)
        save_dict(vars(params), Path(model_dir, "performance.json"))