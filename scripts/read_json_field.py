import json

import fire


def main(json_path: str, field_name: str, sep: str = "."):
    with open(json_path) as f:
        json_data = json.load(f)

    field_name_parts = field_name.split(sep)
    for part in field_name_parts:
        json_data = json_data[part]

    print(json_data, end="")


if __name__ == "__main__":
    fire.Fire(main)
