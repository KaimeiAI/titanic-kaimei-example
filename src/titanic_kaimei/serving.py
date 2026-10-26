"""A scoring service in the shape Kaimei's remote inference calls.

Kaimei sends ``POST {"instances": [row, ...]}`` with each row in the feature schema's own
values (``"Sex": "female"``, ``"Fare": 7.25``) and reads back ``{"predictions": [...]}``: one
``[P(died), P(survived)]`` pair per row, in the order of the model's ``outputs.labels``.

Run it on its own (``pip install "titanic-kaimei-example[serve]"``)::

    python -m titanic_kaimei.serving --port 8080

then register ``https://<your public host>/predict`` as the model's inference URL (notebook 04).
"""

# No ``from __future__ import annotations``: FastAPI reads the request model's annotations at runtime.
import argparse
import socket
import threading
import time
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from .data import COLUMNS_V1, encode, load


def create_app(model: Any, columns: Sequence[str] = COLUMNS_V1, *, api_key: str = "") -> Any:
    """A FastAPI app that scores rows with ``model.predict_proba``.

    ``api_key`` (optional) is required as ``Authorization: Bearer <key>``: Kaimei sends the
    model's ``inference.headers`` with every request.
    """
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel

    class Instances(BaseModel):
        instances: List[Dict[str, Any]]

    app = FastAPI(title="Titanic survival scoring", version="1.0")

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict(body: Instances, authorization: str = Header(default="")) -> Dict[str, Any]:
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="Missing or wrong bearer token.")
        frame = pd.DataFrame(body.instances)
        missing = [name for name in columns if name not in frame]
        if missing:
            raise HTTPException(status_code=422, detail=f"Missing features: {missing}")
        try:
            X = encode(frame[list(columns)])
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"predictions": model.predict_proba(X).tolist()}

    return app


def serve_in_background(app: Any, *, host: str = "127.0.0.1", port: int = 0) -> Tuple[str, Any]:
    """Start ``app`` with uvicorn on a daemon thread; returns its base URL and the server.

    ``port=0`` picks a free port. Stop it with ``server.should_exit = True``.
    """
    import uvicorn

    if port == 0:
        with socket.socket() as probe:
            probe.bind((host, 0))
            port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError(f"The scoring service did not start on {host}:{port}.")
    return f"http://{host}:{port}", server


def train_default_model() -> Any:
    """The XGBoost model of notebook 04, trained on the v1 features."""
    from .models import make_xgboost, split

    titanic = load(1)
    X = encode(titanic.passengers)
    y = (titanic.survived == "survived").astype(int)
    X_train, _, y_train, _ = split(X, y)
    return make_xgboost().fit(X_train, y_train)


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the Titanic XGBoost model for Kaimei's remote inference.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--api-key", default="", help="Require this bearer token on /predict.")
    args = parser.parse_args()
    app = create_app(train_default_model(), api_key=args.api_key)
    print(f"POST http://{args.host}:{args.port}/predict  with  {{\"instances\": [...]}}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
