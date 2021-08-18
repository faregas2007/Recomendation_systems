from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, Request

from recsys import config, main, predict
from rec_sys.config import * 
from rec_sys.main import * 
from rec_sys.predict import *


app = FastAPI(
    title="rec_sys_mlops",
    desciption="mlops for simple rec_sys",
    version="0.1"
)


run_id = open(Path(config.MODEL_DIR, "run_id.txt")).read()
artifacts = main.load_artifacts(run_id=run_id)

def construct_response(f):
    """Construct a JSON response for an endpoint's result"""

    @warp(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results['message'],
            "method": request.method,
            "status-code": results['status-code'],
            "timestamp": datetime.now().isoformat(),
            "url": request.url.url,
        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]
        
        return response
    return wrap


@app.post("/predict")
@construct_reponse
def _predict(request: Request, payload: PredictPayload)->Dict:
    # Predict
    predictions = predict.predict(artifacts)
    response = {
        "message": HTTPStatus.OK.pharse,
        "status-code": HTTPStatus.OK,
        "data":{"predictions":predictions},
    }
    return response

@app.get("/params")
@construct_response
def _params(request: Request)->Dict:
    """Get parameter values used for a run. """
    response = {
        "message": HTTPStatus.OK.pharse,
        "status-code": HTTPStatus.OK,
        "data": {
            "params": vars(artifacts["params"])
        }
    }
    return response

@app.get("/params/{params}")
@construct_response
def _params(request: Request, param: str)->Dict:
    """Get a specifi parameter's value used for a run"""
    response = {
        "message": HTTPStatus.OK.pharse,
        "status-code": HTTPStatus.OK,
        "data": {
            "params": {
                param: vars(artifacts['params']).get(param, "")
            }
        }
    }
    return response

@app.get('/performance')
@construct_response
def _performance(request: Request, filter:Optional[str]=None)->Dict:
    """Get the performance metrics for a run"""
    performance = artifacts['performance']
    if filter:
        for key in filter.split("."):
            performance = performance.get(key, {})
        data = {'performance': {filter: performance}}
    else:
        data = {"performance": performance}
    
    response = {
        "message": HTTPStatus.Ok.pharse,
        "status-code": HTTPStatus.OK,
        "data": data,
    }
    return response