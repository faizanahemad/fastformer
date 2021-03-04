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
import subprocess
import shlex
from distutils.util import strtobool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tabulate import tabulate

from fastformer.utils import one_run, justify


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hosts_file', required=True, type=str,
                        help='Pretrained Model')
    parser.add_argument('-n', '--nodes', default=None,
                        type=str, metavar='N')

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
    parser.add_argument("--gpustat", default=False, action="store_true",
                        help="Flag to do something")
    parser.add_argument('--custom', required=False, type=str, default=None,
                        help='Custom cmd')

    args = parser.parse_args()
    return vars(args)


def run_command_v2(hosts, cmd, args=None, dry_run=False):
    if args is not None:
        args = args[:len(hosts)]
    else:
        args = [None] * len(hosts)

    sep_dict = {"host": [".", 40], "stdout": [" ", 120], "stderr": [" ", 40], "cmd": [" ", 40]}
    with ProcessPoolExecutor(8) as executor:
        ld = list(executor.map(one_run, hosts, [cmd] * len(hosts), args, [dry_run] * len(hosts)))
    if len(ld) == 1:
        ld = ld[0]
        print(ld["host"])
        print(ld["cmd"])
        print(ld["stdout"])
        print(ld["stderr"])
    else:
        # split by \n and then by space
        # "\n".join(["\n".join(list(justify(x.split(sep_dict[key][0]), sep_dict[key][1]))) for x in str(item[key]).split('\n')])
        fns = lambda string, sep, num_chars: "\n".join(["\n".join(list(justify(x.split(sep), num_chars))) for x in str(string).split('\n')])

        dl = {key: [fns(item[key], sep_dict[key][0], sep_dict[key][1]) for item in ld] for key in ld[0].keys()}
        # dl = {key: ["\n".join(list(justify(str(item[key]).split(sep_dict[key][0]), sep_dict[key][1]))) for item in ld] for key in ld[0].keys()}
        print(tabulate(dl, headers="keys", tablefmt="grid"))


if __name__ == "__main__":
    args = get_args()
    hosts = list(pd.read_csv(args["hosts_file"], header=None)[0].values)
    h1 = hosts[:1]
    nodes = args["nodes"]
    if nodes is None:
        pass
    else:
        if "," in nodes or len(nodes.split(":")) == 1:
            if not nodes.startswith("["):
                nodes = "[" + nodes
            if not nodes.endswith("]"):
                nodes = nodes + "]"
        hosts = np.array(hosts)
        hosts = list(eval("hosts["+nodes+"]"))


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
    main_cmd += " --master_addr /home/ahemf/torch_distributed_init --master_port file-9999 --log_every_steps 20 --num_workers 32 --validate_every_steps 40000 --save_every_steps 500"
    main_cmd += " --wandb_dryrun"
    main_cmd += " --resume /home/ahemf/torch_distributed_init/fastformer_checkpoint"
    # main_cmd += " --pretrained_model /home/ahemf/model_save_dir/fastformer.pth"
    # main_cmd += " --validate_on_start"
    main_cmd += " --init_method=file --checkpoint /home/ahemf/torch_distributed_init/fastformer_checkpoint > output.log 2>&1 & disown" # --resume /home/ahemf/torch_distributed_init/fastformer_checkpoint

    # > my.log 2>&1 &
    # cmd0 = "kill -2 $(ps aux | grep train_lm_distributed.py | grep -v grep | awk \'{print $2}\')"
    # cmd1 = "kill -2 $(ps aux | grep multiprocessing | grep -v grep | awk \'{print $2}\')"
    cmd0 = "pkill -9 -f 'train_lm_distributed'"
    cmd1 = "pkill -9 -f 'multiprocessing'"
    cmd2 = "rm ~/torch_distributed_init/file-9999"
    clear_log = cmd_dir + " && rm output.log"
    if args["kill"]:
        run_command_v2(hosts, cmd0)
        run_command_v2(hosts, cmd1)
        run_command_v2(h1, cmd2)
        run_command_v2(hosts, clear_log)
        time.sleep(10)
    if args["ggl"]:
        cmd3 = cmd_dir + " && git pull"
        run_command_v2(hosts, cmd3)
    if args["start"]:
        cmd4 = cmd_dir + " && " + main_cmd
        run_command_v2(hosts, cmd4, list(zip([len(hosts)] * len(hosts), list(map(str, list(range(len(hosts))))))), args["ds"])

    if args["tail"]:
        tail_cmd = cmd_dir + " && tail -n %s output.log" % args["ntail"]
        run_command_v2(hosts, tail_cmd)
    if args["gpustat"]:
        gpustat_cmd = cmd_dir + " && gpustat"
        run_command_v2(hosts, gpustat_cmd)
    if args["custom"] is not None:
        custom_cmd = cmd_dir + " && " + args["custom"]
        run_command_v2(hosts, custom_cmd)








