import itertools
import subprocess
import sys
import os, shutil
from pathlib import Path

import yaml

from .sweep import check_git_detached, process_args_and_load_config


def main():
    check_git_detached()
    config = process_args_and_load_config(local=True)

    parameters = config["parameters"]
    base_command = config["command"]
    parameters = config["parameters"]
    base_command = config["command"]

    args = sys.argv[2:]

    for i, c in enumerate(base_command):
        if c == "${env}":
            base_command[i] = "/usr/bin/env"
        elif c == "${interpreter}":
            base_command[i] = "python -u"
        elif c == "${program}":
            base_command[i] = config["program"]
        elif c == "${args}":
            del base_command[i]

    for k, v in parameters.items():
        parameters[k] = parameters[k]["values"]

    keys, values = zip(*parameters.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    cwd = Path.cwd()
    for i, d in enumerate(permutations_dicts):
        print("\nRunning with configuration:")
        print(yaml.dump(d))
        print()
        command = base_command + [f"--{k}={v}" for k, v in d.items()]
        command = " ".join(command + args)
        print(command)
#        exit()
        result = subprocess.call(command, shell=True)
  
        if result != 0:
            break

        print(f"Current working directory: {cwd}")

#        name_dir="Seed_"+str(i)
        # config_str = json.dumps(d, sort_keys=True)
        # run_id = hashlib.md5(config_str.encode()).hexdigest()[:8]
        # short_name = f"{d['model']}"
        # target_dir = cwd / run_id
        # target_dir.mkdir(parents=True, exist_ok=True)
    
    # save full config
        # with open(folder / "config.json", "w") as f:
        #     json.dump(config, f, indent=2)



# 3. Run cleanup.sh from the command line
# Make sure cleanup.sh is executable (chmod +x cleanup.sh)
        # cleanup_script = cwd/"clean.sh"

        # if cleanup_script.exists():
        #     result1 = subprocess.run(
        #         ["bash", str(cleanup_script)],
        #         cwd=cwd,              # explicitly run from current working directory
        #         capture_output=True,
        #         text=True
        #         )

if __name__ == "__main__":
    main()
