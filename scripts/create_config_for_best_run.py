import os
from pathlib import Path

import wandb

if __name__ == "__main__":
    sweep_path = os.environ["SWEEP_ID"]
    with open("configs/project_name.json") as f:
        import json

        proj_name = json.load(f)["project_name"]

    with open("configs/entity_name.json") as f:
        import json

        entity_name = json.load(f)["entity_name"]

    api = wandb.Api(overrides={"entity": entity_name, "project": proj_name})
    sweep = api.sweep(sweep_path)
    sweep_metric = sweep.config["metric"]["name"]

    best_run = sweep.best_run()

    print(f"Best Run Name: {best_run.name}")
    print(f"Best Run ID: {best_run.id}")
    print(f"Best Run URL: {best_run.url}")
    print(f"Best Run {sweep_metric}: {best_run.summary[sweep_metric]}")

    best_run.file("config.json").download(replace=True)
    Path("config.json").rename("best_run.json")

    with open("best_run_info.json", "w") as f:
        json.dump(
            {
                "name": best_run.name,
                "id": best_run.id,
                "sweep_name": sweep.name,
                "sweep_id": sweep.id,
            },
            f,
        )
