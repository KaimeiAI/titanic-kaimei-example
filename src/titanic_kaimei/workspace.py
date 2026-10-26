"""Small helpers around the Kaimei client that the notebooks share.

Nothing here is required to use Kaimei: each helper is a few calls of the ``kaimei`` package,
kept in one place so the notebooks can focus on what Kaimei shows.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

from kaimei import Kaimei, NotFoundError

PROJECT_SLUG = "titanic"
#: The slugs this example creates (and replaces when a notebook is run again).
FOREST, LOGISTIC, FOREST_V2, XGBOOST, XGBOOST_REMOTE = (
    "titanic-forest",
    "titanic-logistic",
    "titanic-forest-v2",
    "titanic-xgboost",
    "titanic-xgboost-remote",
)


def connect(*, quiet: bool = False) -> Kaimei:
    """A Kaimei client for the key in ``KAIMEI_API_KEY``.

    A ``.env`` file in the working directory or any folder above it is read first (copy
    ``.env.example``). Without a key, an interactive session is asked for one. ``KAIMEI_BASE_URL``
    points at another Kaimei API, for example a local one.
    """
    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:  # python-dotenv is optional
        pass
    if not os.environ.get("KAIMEI_API_KEY") and _can_prompt():
        try:
            from getpass import getpass

            key = getpass("Kaimei API key (dashboard -> API keys): ").strip()
        except Exception:  # no stdin, e.g. a notebook run headless
            key = ""
        if key:
            os.environ["KAIMEI_API_KEY"] = key
    if not os.environ.get("KAIMEI_API_KEY"):
        raise RuntimeError(
            "Set KAIMEI_API_KEY (or put it in a .env file) to a key from https://dashboard.kaimei.ai -> API keys."
        )
    client = Kaimei()
    if not quiet:
        organization = client.organization.retrieve()
        api = os.environ.get("KAIMEI_BASE_URL") or "https://api.kaimei.ai"
        print(f"Connected to Kaimei at {api} (workspace: {organization.name})")
    return client


def _can_prompt() -> bool:
    """A person can type a key: a Jupyter kernel, or a terminal."""
    if "ipykernel" in sys.modules:
        return True
    return bool(sys.stdin and sys.stdin.isatty())


def dashboard_url(path: str = "") -> str:
    """A link into the Kaimei dashboard (``KAIMEI_DASHBOARD_URL`` overrides the default)."""
    base = os.environ.get("KAIMEI_DASHBOARD_URL", "https://dashboard.kaimei.ai").rstrip("/")
    return f"{base}/{path.lstrip('/')}" if path else base


def ensure_project(client: Kaimei) -> Any:
    """The ``titanic`` project, created the first time."""
    try:
        return client.projects.retrieve(PROJECT_SLUG)
    except NotFoundError:
        return client.projects.create(
            name="Titanic survival",
            slug=PROJECT_SLUG,
            description="Who survived the Titanic? Models trained on the Kaggle passenger list "
            "(github.com/KaimeiAI/titanic-kaimei-example).",
        )


def delete_if_exists(client: Kaimei, slug: str) -> bool:
    """Delete one of this example's models so a notebook can be run again from scratch."""
    try:
        client.models.delete(slug)
        return True
    except NotFoundError:
        return False


def replace_model(client: Kaimei, estimator: Any, *, slug: str, **upload: Any) -> Any:
    """``client.models.upload`` after deleting a previous upload with the same slug."""
    delete_if_exists(client, slug)
    return client.models.upload(estimator, slug=slug, **upload)


def run_analysis(client: Kaimei, model: str, kind: str = "full", *, seed: Optional[int] = 0) -> Dict[str, Any]:
    """Run an analysis, wait for it, and return the model's latest insight of each type."""
    options = {"seed": seed} if seed is not None else {}
    analysis = client.analyses.wait(client.analyses.create(model=model, type=kind, options=options).id, timeout=600)
    if analysis.status != "completed":
        raise RuntimeError(f"Analysis {analysis.id} {analysis.status}: {analysis.get('error')}")
    return {insight.type: insight for insight in client.insights.list(model=model).data}
