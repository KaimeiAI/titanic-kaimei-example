# Titanic × Kaimei

A worked example of [Kaimei](https://kaimei.ai) on a familiar model: who survived the Titanic?

[`notebooks/titanic-guide-with-sklearn-and-eda.ipynb`](notebooks/titanic-guide-with-sklearn-and-eda.ipynb) is a classic Kaggle notebook. It cleans the passenger list, engineers features and compares eight scikit-learn classifiers by accuracy.

[`notebooks/kaimei-demo.ipynb`](notebooks/kaimei-demo.ipynb) picks up where accuracy stops. It trains a random forest and a logistic regression on the same features, uploads both to Kaimei, and asks why they decide what they decide: in Python, and on the Kaimei dashboard.

## What the demo shows

| Question | Where | Kaimei call |
| --- | --- | --- |
| Why did the model predict this for this person? | Notebook §5, dashboard **Explain** tab | `client.explanations.create(..., method="shap")` |
| What is the smallest change that flips the decision? | Notebook §6 | `options={"counterfactual": True, "frozen_features": [...]}` |
| Do two models agree, and why not? | Notebook §7 | the same explanation on both models |
| What does the model rely on overall, and what has it learned? | Notebook §8, dashboard **Insights** | `client.analyses.create(model, type="full")` |
| Is production data drifting away from the training data? | Notebook §8, dashboard **Insights** | `client.predictions.create_batch(...)` + the drift insight |
| What does the model look like inside? | Dashboard **Overview** (architecture card) | uploaded by `client.models.upload(...)` |

Along the way it turns up things accuracy never shows:
- The logistic regression can't use the label-encoded `Title`.
- The forest has partly learned the median age used to fill missing ages.
- The Kaggle `Fare` is a per-ticket price, which interacts with family size.

## Run it

```bash
pip install -r requirements.txt
export KAIMEI_API_KEY=km_live_...   # dashboard → API keys (PowerShell: $env:KAIMEI_API_KEY = "km_live_...")
jupyter lab notebooks/kaimei-demo.ipynb
```

Run all cells (about a minute). The notebook creates a **Titanic survival** project with two models, `titanic-forest` and `titanic-logistic`, and prints their dashboard links. Running it again replaces those two models.

Optional settings:
- `KAIMEI_BASE_URL`: point at another Kaimei API, for example a local one.
- `KAIMEI_DASHBOARD_URL`: change the printed links.

The notebook needs the `kaimei` Python client 0.6.0 or later; that release adds the `kaimei.tabular` plots. Until it's on PyPI, install the client from a checkout: `pip install -e "path/to/kaimei-python-client[models]"`.

## Suggested demo flow (15 minutes)

1. **The problem (1 min).** Open the original notebook's model table: eight models, one accuracy column, no reasons.
2. **Upload (2 min).** Notebook §3–4. The point to make is that the schema is the contract: the dashboard and every explanation read `Sex = female` and `Fare = 7.25 GBP`, and `upload` is one call per model. Then open the printed dashboard link.
3. **Dashboard (4 min).**
   - On the forest's **Overview**, walk Mr Braund down one of the 30 trees.
   - On **Explain**, start from the pre-filled typical passenger, change `Sex` to `female`, and explain again.
   - On the project page, show both models side by side.
4. **One passenger, in depth (4 min).** Notebook §5–7: the waterfall, the "which ticket would have saved him?" counterfactual, and the two models disagreeing about him.
5. **The whole model (3 min).** Notebook §8: importance, learned rules (the age-28 artefact), interactions (fare × family size), and drift on the logged traffic. Then the same cards on the dashboard's **Insights** tab.
6. **Your models (1 min).** Notebook §9: what changes for PyTorch, Keras, XGBoost or a model behind an API (very little).

## Using Kaimei with your own models

```python
from kaimei import Kaimei

client = Kaimei()  # KAIMEI_API_KEY from the environment
model = client.models.upload(fitted_model, name="Churn v4", features=FEATURES, X=X_reference, y=y_reference)
explanation = client.explanations.create(model=model.slug, input=row, method="shap")
```

- **Run by Kaimei** (explanations and analyses need nothing else from you): scikit-learn linear models, trees, forests of up to 30 trees, MLPs and `StandardScaler` pipelines; Keras / TensorFlow dense and convolutional networks; PyTorch modules, including large ones.
- **Anything else** (XGBoost, LightGBM, an SVC, a model behind your own service): register it with `inference={"type": "remote", "url": ...}` and Kaimei calls your endpoint. Alternatively, post attributions you already compute.
- **Weights that must stay private**: `weights="summary"` uploads shapes only, and scoring goes through your endpoint.

The [Python client README](https://github.com/KaimeiAI/kaimei-python-client#readme) covers each of these. Every call is also plain REST (`https://api.kaimei.ai/v1/...`).

## Data

`data/train.csv` and `data/test.csv` are the [Kaggle Titanic](https://www.kaggle.com/competitions/titanic/data) files. The original notebook reads them from Kaggle's `../input/` folder; the Kaimei demo reads `data/`.
