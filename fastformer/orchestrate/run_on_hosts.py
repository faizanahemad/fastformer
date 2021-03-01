# pwd
# kill -2 $(ps aux | grep multiprocessing | grep -v grep | awk '{print $2}')
# kill -2 $(ps aux | grep train_lm_distributed.py | grep -v grep | awk '{print $2}')
# rm ~/torch_distributed_init/file-9999
# cd /home/ahemf/mygit/fastformer/fastformer/training
# git pull
#

import pandas as pd
import random
import os
import argparse
import numpy as np
from tqdm.auto import tqdm
from pssh.clients import ParallelSSHClient


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hosts_file', required=True, type=str,
                        help='Pretrained Model')
    parser.add_argument('-n', '--nodes', default=None,
                        type=int, metavar='N')
    args = parser.parse_args()
    return vars(args)


def run_command(hosts, nodes, cmd, args=None):
    client = ParallelSSHClient(hosts[:nodes])
    output = client.run_command(cmd, host_args=args, shell="zsh")
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
    test_cmd1 = "pwd"
    test_cmd2 = "echo $USER"
    test_cmd3 = "echo $SHELL"
    test_cmd4 = "echo $0"
    cmd0 = "kill -2 $(ps aux | grep train_lm_distributed.py | grep -v grep | awk '{print $2}')"
    cmd1 = "kill -2 $(ps aux | grep multiprocessing | grep -v grep | awk '{print $2}')"
    cmd2 = "rm ~/torch_distributed_init/file-9999"
    cmd_dir = "cd /home/ahemf/mygit/fastformer/fastformer/training"
    main_cmd = """nohup python train_lm_distributed.py -n 8 -g 8 --nr %s --model_config md_config --model_save_dir /home/ahemf/model_save_dir --model_save_name fastformer.pth --train_dataset /home/ahemf/processed_datasets/train_fastformer_resampled_10M --validation_dataset /home/ahemf/processed_datasets/validation_fastformer --master_addr /home/ahemf/torch_distributed_init --master_port file-9999 --log_every_steps 50 --cpu False --num_workers 32 --validate_every_steps 5000 --pretrained_model /home/ahemf/model_save_dir/fastformer.pth --init_method=file &"""
    cmd3 = cmd_dir + " && git pull"
    cmd4 = cmd_dir + " && " + main_cmd
    run_command(hosts, nodes, test_cmd1)
    run_command(hosts, nodes, test_cmd2)
    run_command(hosts, nodes, test_cmd3)
    run_command(hosts, nodes, test_cmd4)
    #
    # run_command(hosts, nodes, cmd0)
    # run_command(hosts, nodes, cmd1)
    # run_command(hosts, nodes, cmd2)
    # run_command(hosts, nodes, cmd3)
    # run_command(hosts, nodes, cmd4, list(map(str, list(range(nodes)))))







