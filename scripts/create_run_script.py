import os
from pathlib import Path

import fire


def add_python_paths():
    script = (
        "\n\nexport PYTHONPATH=$HOME/.local/lib/python3.6/site-packages/:$PYTHONPATH\n"
    )
    script += (
        "export PYTHONPATH=$HOME/.local/lib/python3.7/site-packages/:$PYTHONPATH\n"
    )
    script += (
        "export PYTHONPATH=$HOME/.local/lib/python3.8/site-packages/:$PYTHONPATH\n"
    )
    script += (
        "export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages/:$PYTHONPATH\n"
    )
    script += "\n\n#pip install --user -r src/requirements.txt\n"
    return script


def make_executable(script_path):
    mode = os.stat(str(script_path)).st_mode
    mode |= (mode & 0o444) >> 2
    os.chmod(str(script_path), mode)


def main(exp_dir: str, output_path: str):
    exp_dir = Path(exp_dir)
    config_path = exp_dir / "config.json"
    with open(config_path) as f:
        import json

        config = json.load(f)

    _ = config.pop("config_filenames", None)
    _ = config.pop("sweep_run", None)

    exp_name = config["exp_name"]

    new_config_path = Path(f"cfg_{exp_dir.name}.json")
    with open(new_config_path, "w") as f:
        json.dump(config, f, indent=4)

    script_str = "#!/bin/bash\n\n"
    script_str += add_python_paths()
    script_str += f"CONFIG_PATH={new_config_path}\n\n"
    script_str += f"export APP_EXPERIMENT_NAME={exp_name}\n\n"
    script_str += f"python src/main.py --debug_mode --configs $CONFIG_PATH \\\n"
    script_str += f"       train\n\n"

    with open(output_path, "w") as f:
        f.write(script_str)

    make_executable(output_path)

    print(f"New config path: {new_config_path}")
    print(f"Created script at {output_path}")

if __name__ == "__main__":
    fire.Fire(main)
