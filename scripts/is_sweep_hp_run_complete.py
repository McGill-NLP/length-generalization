import json
from pathlib import Path

import fire
import wandb


def is_run_complete(run: wandb.apis.public.Run) -> bool:
    # Assert training is done
    try:
        max_steps = run.config["trainer"]["max_steps"]
    except Exception as e:
        return False

    h = run.history(samples=500, keys=["train/loss"], x_axis="train/global_step")
    if h is None or len(h) == 0:
        return False
    last_step = h.iloc[-1]["train/global_step"]

    if abs(last_step - max_steps) > 0.01 * max_steps:
        return False

    # Assert validation is done
    if (
        "pred/valid_acc_overall" not in run.summary
        or run.summary["pred/valid_acc_overall"] is None
    ):
        return False

    return True


def is_training_complete(exp_dir: Path):
    try:
        config = json.load((exp_dir / "config.json").open())
        max_steps = config["trainer"]["max_steps"]
    except Exception as e:
        return False

    try:
        trainer_state = json.load(
            (exp_dir / "checkpoints" / "trainer_state.json").open()
        )
        last_step = trainer_state["global_step"]
    except Exception as e:
        return False

    if abs(last_step - max_steps) > 0.01 * max_steps:
        return False

    return True


def is_eval_complete(exp_dir: Path, metric_name: str) -> bool:
    analysis_root = exp_dir / "analysis"
    if not analysis_root.exists():
        return False

    is_complete = False
    for analyzer_dir in analysis_root.iterdir():
        if not analyzer_dir.is_dir():
            continue

        log_file = analyzer_dir / "log.json"
        if not log_file.exists():
            continue

        try:
            log = json.load(log_file.open())
            if metric_name in log:
                is_complete = True
                break
        except Exception as e:
            continue

    return is_complete


def main(experiment_dir: str, metric_name: str):
    experiment_dir = Path(experiment_dir)

    if not is_training_complete(experiment_dir):
        print("False", end="")
        return

    if not is_eval_complete(experiment_dir, metric_name):
        print("False", end="")
        return

    print("True", end="")


if __name__ == "__main__":
    fire.Fire(main)
