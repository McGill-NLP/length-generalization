import json
from pathlib import Path
from typing import List, Dict, Any


def modify_array(arr: List[Any], idx: int, val: Any) -> List[Any]:
    arr[idx] = val
    return arr


def generate_boolean_configs(config):
    configs = []
    for i in range(len(config)):
        new_config = config.copy()
        new_config[i] = not config[i]
        configs.append(new_config)
    return configs


def generate_all_boolean_combinations(length: int) -> List[List[bool]]:
    combs = []
    for i in range(2**length):
        comb = []
        for j in range(length):
            comb.append(bool(i & (1 << j)))
        combs.append(comb)

    return combs


def generate_all_scratchpad_configs() -> List[Dict[str, Any]]:
    configs = []
    scratchpad_bool_config = [
        True,  # include_input
        True,  # include_computation
        True,  # include_output
        True,  # include_intermediate_variables
        True,  # include_remaining_input
    ]
    all_bool_configs = generate_all_boolean_combinations(len(scratchpad_bool_config))

    for config in all_bool_configs:
        configs.append(
            {
                "include_input": config[0],
                "include_computation": config[1],
                "include_output": config[2],
                "include_intermediate_variables": config[3],
                "include_remaining_input": config[4],
            }
        )

    return configs


def get_file_name(config: Dict[str, bool]) -> str:
    keys_to_abrv = {
        "include_input": "i",
        "include_computation": "c",
        "include_output": "o",
        "include_intermediate_variables": "v",
        "include_remaining_input": "r",
    }
    keys_in_order = [
        "include_input",
        "include_computation",
        "include_output",
        "include_intermediate_variables",
        "include_remaining_input",
    ]
    filename = ""
    for key in keys_in_order:
        abrv = keys_to_abrv[key]
        val = str(int(config[key]))
        filename += f"{abrv}{val}_"

    return filename[:-1]


JSONNET_CONFIG_TEMPLATE = """{
    dataset+: {
        instance_processor+: {},
    },
}
"""


def main():
    configs_dir = (
        Path(__file__).parent.parent / "configs" / "data" / "unified_scratchpad_configs"
    )
    configs_dir.mkdir(parents=True, exist_ok=True)
    for config in generate_all_scratchpad_configs():
        filename = get_file_name(config)
        with (configs_dir / f"ufs__{filename}.jsonnet").open("w") as f:
            config_str = json.dumps(config, indent=4)
            f.write(JSONNET_CONFIG_TEMPLATE.replace("{}", config_str))


if __name__ == "__main__":
    main()
