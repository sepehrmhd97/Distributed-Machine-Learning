import ray
from ray import tune
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Load the forest cover type dataset
covtype = fetch_covtype()
X = covtype.data[:10000]
y = covtype.target[:10000]

# Define the hyperparameters to tune
config = {
    "max_depth": tune.grid_search([10, 20, 30, None]),
    "n_estimators": tune.grid_search([50, 100, 150]),
    "ccp_alpha": tune.grid_search([0.0, 0.001, 0.01])
}

# Define the training function
def train_rf(config):
    # Create a random forest classifier with the given hyperparameters
    rf = RandomForestClassifier(
        max_depth=config["max_depth"],
        n_estimators=config["n_estimators"],
        ccp_alpha=config["ccp_alpha"],
        n_jobs=-1
    )

    # Evaluate the classifier using cross-validation
    scores = cross_val_score(rf, X, y, cv=5, n_jobs=-1)
    mean_score = np.mean(scores)

    # Return the mean cross-validation score as the result
    return {"mean_cv_score": mean_score}

# Configure Ray Tune
ray.init()
analysis = tune.run(
    train_rf,
    config=config,
    num_samples=9,
    scheduler=tune.schedulers.ASHAScheduler(metric="mean_cv_score", mode="max"),
    resources_per_trial={"cpu": 1},
    verbose=1
)

# Print the best hyperparameters and their associated score
best_config = analysis.get_best_config(metric="mean_cv_score", mode="max")
best_score = analysis.best_result["mean_cv_score"]
print(f"Best config: {best_config}, best score: {best_score}")
