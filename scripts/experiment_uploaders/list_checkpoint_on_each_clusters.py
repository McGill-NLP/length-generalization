import os
from collections import defaultdict

import paramiko


def list_available_models_groups(hostname, username, remote_path):
    """
    List a directory recursively in a remote location using SSH with password-less login and SSH configuration.
    Args:
        hostname (str): Hostname of the remote server.
        username (str): Username for SSH authentication.
        remote_path (str): Remote path of the directory to list.
    Returns:
        list: A list of file paths in the remote directory, including subdirectories.
    """
    ssh_config = paramiko.SSHConfig()
    ssh_config.parse(open(os.path.expanduser("~/.ssh/config"), "r"))
    ssh_config_host = ssh_config.lookup(hostname)
    ssh_client = paramiko.SSHClient()
    ssh_client.load_system_host_keys()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(
        hostname=ssh_config_host["hostname"],
        username=username or ssh_config_host["user"],
        port=int(ssh_config_host.get("port", "22")),
        key_filename=ssh_config_host.get("identityfile"),
        look_for_keys=True,
    )
    sftp_client = ssh_client.open_sftp()

    remote_path = os.path.join(remote_path, "scratch", "len_gen", "experiments")

    output_dict = defaultdict(list)

    for item in sftp_client.listdir_attr(remote_path):
        if not item.filename.startswith("SW-"):
            continue

        try:
            for item2 in sftp_client.listdir_attr(
                os.path.join(remote_path, item.filename, "exps")
            ):
                if not item2.filename.startswith("best_run_seed_"):
                    continue
                seed = item2.filename
                has_checkpoint = False
                for item3 in sftp_client.listdir_attr(
                    os.path.join(
                        remote_path, item.filename, "exps", seed, "checkpoints"
                    )
                ):
                    if item3.filename.endswith("pytorch_model.bin"):
                        has_checkpoint = True
                        break
                if has_checkpoint:
                    output_dict[item.filename].append(seed)

        except FileNotFoundError:
            continue

    sftp_client.close()
    ssh_client.close()
    return output_dict


if __name__ == "__main__":
    cedar_groups = list_available_models_groups("cc-cedar-5", None, "/home/kzmnjd/")
    print("Len of cedar groups:", len(cedar_groups))
    narval_groups = list_available_models_groups("cc-narval-4", None, "/home/kzmnjd/")
    print("Len of narval groups:", len(narval_groups))

    # mila_groups = list_available_models_groups(
    #     "mila4", None, "/home/mila/a/amirhossein.kazemnejad/"
    # )
    # print("Len of mila groups:", len(mila_groups))
    mila_groups = {}


    # Save the output to a json file
    import json

    # Make a backup of the current file if it exists
    if os.path.exists("available_groups.json"):
        # Rename it with a timestamp postfix
        import time
        os.rename(
            "available_groups.json",
            f"available_groups_{time.strftime('%Y%m%d-%H%M%S')}.json",
        )

    with open("available_groups.json", "w") as f:
        json.dump(
            {"mila": mila_groups, "cedar": cedar_groups, "narval": narval_groups},
            f,
            indent=4,
        )