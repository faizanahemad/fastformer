# pwd
# kill -2 $(ps aux | grep multiprocessing | grep -v grep | awk '{print $2}')
# kill -2 $(ps aux | grep train_lm_distributed.py | grep -v grep | awk '{print $2}')
# rm ~/torch_distributed_init/file-9999
# cd /home/ahemf/mygit/fastformer/fastformer/training
# git pull
#
import time

import pandas as pd
import random
import os
import argparse
import numpy as np
from tqdm.auto import tqdm
from pssh.clients import ParallelSSHClient, SSHClient
import subprocess
import shlex
from distutils.util import strtobool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# TODO: Parallel with multi-thread

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hosts_file', required=True, type=str,
                        help='Pretrained Model')
    parser.add_argument('-n', '--nodes', default=None,
                        type=int, metavar='N')

    parser.add_argument("--start", default=False, action="store_true",
                        help="Flag to do something")
    parser.add_argument("--ggl", default=False, action="store_true",
                        help="Flag to do something")
    parser.add_argument("--ds", default=False, action="store_true",
                        help="Flag to do something")
    parser.add_argument("--tail", default=False, action="store_true",
                        help="Flag to do something")
    parser.add_argument('--ntail', required=False, type=int, default=20,
                        help='Tail Length')
    parser.add_argument("--kill", default=False, action="store_true",
                        help="Flag to do something")

    args = parser.parse_args()
    return vars(args)


def one_run(host, cmd, arg, dry_run=False):
    cur_cmd = (cmd % arg) if arg is not None else cmd
    if dry_run:
        return {"host": host, "cmd": cur_cmd, "stdout": "", "stderr": ""}
    cmd_str = shlex.split("ssh %s '%s'" % (host, cur_cmd))
    s = subprocess.run(cmd_str, shell=False, capture_output=True, text=True)
    return {"host": host, "stdout": s.stdout, "stderr": s.stderr, "cmd": cur_cmd}


def run_command_v2(hosts, nodes, cmd, args=None, dry_run=False):
    hosts = hosts[:nodes]
    if args is not None:
        args = args[:nodes]
    else:
        args = [None] * len(hosts)

    with ProcessPoolExecutor(8) as executor:
        for results in executor.map(one_run, hosts, [cmd] * len(hosts), args, [dry_run] * len(hosts)):
            print(results["host"])
            print(results["cmd"])
            print(results["stdout"], results["stderr"])
            print("#" * 80)


def run_command(hosts, nodes, cmd, args=None):
    if nodes > 1:
        client = ParallelSSHClient(hosts[:nodes], pkey="~/.ssh/id_rsa", password="")
    else:
        client = SSHClient(hosts[0], pkey="~/.ssh/id_rsa", password="")
    output = client.run_command(cmd, host_args=args[:nodes], shell="zsh")
    client.join()
    for host_output in output:
        hostname = host_output.host
        stdout = list(host_output.stdout)
        print("Host %s: exit code %s, output %s" % (
            hostname, host_output.exit_code, stdout))


if __name__ == "__main__":
    args = get_args()
    hosts = list(pd.read_csv(args["hosts_file"], header=None)[0].values)
    nodes = args["nodes"]
    if nodes is None:
        nodes = len(hosts)
    # test_cmd1 = "pwd"
    # test_cmd2 = "echo $USER"
    # test_cmd3 = "echo $SHELL"
    # test_cmd4 = "echo $0"
    # test_cmd5 = "which python"
    # test_cmd6 = "source ~/.zshrc && which python"
    #
    # run_command_v2(hosts, nodes, test_cmd1)
    # run_command_v2(hosts, nodes, test_cmd2)
    # run_command_v2(hosts, nodes, test_cmd3)
    # run_command_v2(hosts, nodes, test_cmd4)
    # run_command_v2(hosts, nodes, test_cmd5)
    # run_command_v2(hosts, nodes, test_cmd6)
    #

    cmd_dir = "source ~/.zshrc && cd /home/ahemf/mygit/fastformer/fastformer/training"
    main_cmd = """python train_lm_distributed.py -n %s -g 8 --nr %s --model_config md_config"""
    main_cmd += " --model_save_dir /home/ahemf/model_save_dir --model_save_name fastformer.pth"
    main_cmd += " --train_dataset /home/ahemf/processed_datasets/train_fastformer_resampled_10M --validation_dataset /home/ahemf/processed_datasets/validation_fastformer"
    main_cmd += " --master_addr /home/ahemf/torch_distributed_init --master_port file-9999 --log_every_steps 20 --cpu False --num_workers 32 --validate_every_steps 10000 --save_every_steps 500"
    main_cmd += " --wandb_dryrun"
    main_cmd += " --resume /home/ahemf/torch_distributed_init/fastformer_checkpoint"
    main_cmd += " --pretrained_model /home/ahemf/model_save_dir/fastformer.pth"
    main_cmd += " --init_method=file --checkpoint /home/ahemf/torch_distributed_init/fastformer_checkpoint > output.log 2>&1 & disown" # --resume /home/ahemf/torch_distributed_init/fastformer_checkpoint
    # > my.log 2>&1 &
    # cmd0 = "kill -2 $(ps aux | grep train_lm_distributed.py | grep -v grep | awk \'{print $2}\')"
    # cmd1 = "kill -2 $(ps aux | grep multiprocessing | grep -v grep | awk \'{print $2}\')"
    cmd0 = "pkill -9 -f 'train_lm_distributed'"
    cmd1 = "pkill -9 -f 'multiprocessing'"
    cmd2 = "rm ~/torch_distributed_init/file-9999"
    clear_log = cmd_dir + " && rm output.log"
    if args["kill"]:
        run_command_v2(hosts, nodes, cmd0)
        run_command_v2(hosts, nodes, cmd1)
        run_command_v2(hosts[:1], 1, cmd2)
        run_command_v2(hosts, nodes, clear_log)
        time.sleep(10)
    if args["ggl"]:
        cmd3 = cmd_dir + " && git pull"
        run_command_v2(hosts, nodes, cmd3)
    if args["start"]:
        cmd4 = cmd_dir + " && " + main_cmd
        run_command_v2(hosts, nodes, cmd4, list(zip([nodes] * len(hosts), list(map(str, list(range(nodes)))))), args["ds"])

    if args["tail"]:
        tail_cmd = cmd_dir + " && tail -n %s output.log" % args["ntail"]
        run_command_v2(hosts, nodes, tail_cmd)









