import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli import app
from recsys import utils, config

runner = CliRunner()

@pytest.mark.training
def test_optimize():
    study_name = "test_optimization"
    result = runner.invoke(app,
        [
            "optimize",
            "--params-fp",
            f"{Path(config.config_dir, 'test_params.json')}",
            "--study-name",
            f"{study_name}",
            "--num_trials",
            1,
        ],
    )

    assert result.exit_code == 0
    assert "Trial 0" in result.stdout

    # delete study
    utils.delete_experiment(experiment_name=study_name)
    shutil.rmtree(Path(config.model_registry, ".trash"))