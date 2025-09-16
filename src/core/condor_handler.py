import os
import sys

from src.classes.config_class import SSConfig

ss_config = SSConfig()


def make_htcondor(out_dir, queue):
    """
    Make the htcondor file for the condor submission
    -----------------------------------------------
    Args:
        out_dir: output directory
        queue: queue to submit to
    -----------------------------------------------
    Returns:
        htcondor: htcondor file
        done: file to indicating if job is done
    -----------------------------------------------
    """

    lines = []
    lines.append("executable = " + out_dir + "/htcondor.sh\n")
    lines.append("output = " + out_dir + "/" + os.path.basename(out_dir) + ".out\n")
    lines.append("error = " + out_dir + "/" + os.path.basename(out_dir) + ".err\n")
    lines.append("log = " + out_dir + "/" + os.path.basename(out_dir) + ".log\n")
    lines.append('+JobFlavour = "' + queue + '"\n')
    lines.append("RequestCpus = 12\n")
    lines.append("queue 1")

    htcondor = out_dir + "/htcondor"

    f = open(htcondor, "a")
    f.writelines(lines)
    f.close()
    return htcondor, out_dir + "/" + os.path.basename(out_dir) + "-done"


def make_script(cmd, script, done):
    """
    Make the script to be run on the condor node
    -----------------------------------------------
    Args:
        cmd: command to be run
        script: script to be run
        done: file to indicating if job is done
    -----------------------------------------------
    Returns:
        None
    -----------------------------------------------
    """

    lines = []
    username = os.environ["USER"]
    lines.append("#!/usr/bin/bash\n")
    lines.append(f"source /afs/cern.ch/user/{username[0]}/{username}/.bashrc\n")
    lines.append("cd " + os.getcwd() + "\n")
    lines.append(f"conda activate scales-env\n")
    lines.append("\n")
    lines.append(f"python {cmd} --from-condor\n")
    lines.append("\n")
    lines.append("touch " + done)

    f = open(script + ".sh", "a")
    f.writelines(lines)
    f.close()


def manage(cmd, out, queue):
    """
    Manage the condor submission
    -----------------------------------------------
    Args:
        cmd: command to be run
        out: output directory
        queue: queue to submit to
    -----------------------------------------------
    Returns:
        None
    -----------------------------------------------
    """

    target_dir = f"{os.getcwd()}/condor/{out}"

    # check for proper directory structure
    if not os.path.exists(target_dir):
        os.system("mkdir -p " + target_dir)
    else:
        print(
            f"[ERROR][python/condor_handler][manage] The directory {target_dir} already exists"
        )
        print(
            "[INFO][python/condor_handler][manage] to continue this directory will be formatted."
        )
        cont = input("[INFO][python/condor_handler][manage] continue? [Y/n]:\n")
        if cont == "Y":
            os.system("rm -rf " + target_dir)
            os.system("mkdir -p " + target_dir)
        elif cont == "n":
            print("[EXIT]")
            return
        else:
            print("[ERROR][python/condor_handler][manage] could not parse input")
            print("[EXIT]")
            return

    htcondor, done = make_htcondor(target_dir, queue)
    make_script(cmd, htcondor, done)

    condor_submit = "condor_submit --batch-name " + out + " " + htcondor
    print("[INFO][python/condor_handler][manage] submitting job to condor")
    print(condor_submit)
    os.system(condor_submit)
