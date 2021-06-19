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
    parser.add_argument("--lm", default=False, action="store_true",
                        help="Flag to do something")
    parser.add_argument("--vision", default=False, action="store_true",
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
    parser.add_argument('--scp', required=False, type=str, default=None,
                        help='scp cmd in format `scp -rC <from> <uname>@%s:<to>`')

    args = parser.parse_args()
    return vars(args)


def run_command_v2(hosts, cmd, args=None, dry_run=False):
    import re
    import shutil
    if args is not None:
        args = args[:len(hosts)]
    else:
        args = [None] * len(hosts)

    tsize = shutil.get_terminal_size()[0]
    with ThreadPoolExecutor(min(64, len(hosts))) as executor:  # ProcessPoolExecutor is quite slower
        ld = list(executor.map(one_run, hosts, [cmd] * len(hosts), args, [dry_run] * len(hosts)))
    if len(ld) <= 2:
        for ll in ld:

            print(ll["host"])
            print(ll["cmd"])
            if len(ll["stdout"].strip()) > 0:
                print(ll["stdout"])
            if len(ll["stderr"].strip()) > 0:
                print(ll["stderr"])
            print("-" * (tsize - 10))
    elif dry_run:
        _ = [print(ll["cmd"]) for ll in ld]
    else:
        # split by \n and then by space
        # "\n".join(["\n".join(list(justify(x.split(sep_dict[key][0]), sep_dict[key][1]))) for x in str(item[key]).split('\n')])
        fns = lambda string, sep, num_chars: "\n".join(["\n".join(list(justify(list(x), num_chars))) for x in str(string).split('\n')])

        tsize = shutil.get_terminal_size()[0]
        _ = [out.pop("cmd", None) for out in ld]
        if sum([len(out["stderr"].strip()) for out in ld]) == 0:
            _ = [out.pop("stderr", None) for out in ld]
            stderr_size = 0.0
            stdout_size = 0.7
        else:
            stderr_size = 0.3
            stdout_size = 0.4
        sep_dict = {"host": [".", int(0.3 * tsize)], "stdout": [" ", int(stdout_size * tsize)], "stderr": [" ", int(stderr_size * tsize)]}

        dl = {key: [fns(item[key], sep_dict[key][0], sep_dict[key][1]) for item in ld] for key in ld[0].keys()}
        # dl = {key: ["\n".join(list(justify(str(item[key]).split(sep_dict[key][0]), sep_dict[key][1]))) for item in ld] for key in ld[0].keys()}

        dl["id"] = list(range(len(list(dl.values())[0])))
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
    main_cmd = """python train_lm_distributed.py -n %s -g 8 --nr %s --model_config tg_config"""
    main_cmd += " --model_save_dir /home/ahemf/model_save_dir --model_save_name fastformer.pth"

    # main_cmd += " --train_dataset /home/ahemf/processed_datasets/train_fastformer_resampled_10M"
    # main_cmd += " --train_dataset /home/ahemf/processed_datasets/train_fastformer_resampled_50M"
    main_cmd += " --train_dataset /home/ahemf/processed_datasets/train_fastformer_resampled_100M"
    # main_cmd += " --train_dataset /home/ahemf/processed_datasets/train_qna_medium_fastformer_5M"
    # main_cmd += " --train_dataset /home/ahemf/processed_datasets/train_qna_small_fastformer_5M"

    main_cmd += " --validation_dataset /home/ahemf/processed_datasets/validation_fastformer"
    main_cmd += " --log_every_steps 100 --num_workers 8 --validate_every_steps 100000 --save_every_steps 5000"
    # main_cmd += " --wandb_dryrun"

    # main_cmd += " --init_method=file --master_addr /home/ahemf/torch_distributed_init --master_port file-9999"
    main_cmd += " --init_method=tcp --master_addr 172.19.171.55 --master_port 9999"

    # main_cmd += " --resume /home/ahemf/torch_distributed_init/fastformer_checkpoint"

    main_cmd += " --pretrained_model /home/ahemf/model_save_dir/fastformer.pth"

    # main_cmd += " --validate_on_start --validate_only"

    # main_cmd += " --checkpoint /home/ahemf/torch_distributed_init/fastformer_checkpoint"

    # main_cmd += " --cpu"
    main_cmd += " --no_autocast"
    # main_cmd += " --detect_anomaly"
    main_cmd += " --accumulation_steps 2" #  4096/(16*8*8)
    main_cmd += " --shuffle_dataset"
    # main_cmd += " --backward_hook"

    main_cmd += " > output.log 2>&1 & disown"

    vision_cmd = "python train_vision_distributed.py -n %s -g 8 --nr %s --model_config vision_base_config"
    vision_cmd += " --log_every_steps 100 --num_workers 8 --save_every_steps 2000 --init_method=tcp  --master_addr 0.0.0.0 --master_port 9999 --shuffle_dataset"
    vision_cmd += " --accumulation_steps 1"
    vision_cmd += " --no_autocast"

    vision_cmd += " --epochs 100"
    vision_cmd += " --wandb_name val_1_1_1_1_32"
    vision_cmd += " --model_save_dir /home/ahemf/model_save_dir --model_save_name imagenet.pth --dataset /home/ahemf/processed_datasets/ImageNet"
    vision_cmd += " --pretrained_model /home/ahemf/model_save_dir/patchclr.pth"
    vision_cmd += " --mode linear_probe"
    vision_cmd += " --lr 0.002 --lr_steps 2 --weight_decay 0.05 --warmup_steps 10000 --gradient_clipping 3.0 --simclr_warmup 1000 --teacher_warmup_steps 1000"
    vision_cmd += " --batch_size 32 --optimizer adamw"
    # vision_cmd += " --simclr_w 0.0 --generator_w 1.0 --discriminator_w 1.0 --dino_w 0.0"
    # vision_cmd += " --detect_anomaly"

    # vision_cmd += " --wandb_dryrun"
    # vision_cmd += " --deit"
    vision_cmd += " > output.log 2>&1 & disown"

    lm_cmd = "python train_mtt_distributed.py -n %s -g 8 --nr %s --model_save_dir /home/ahemf/model_save_dir --model_save_name mtt.pth"
    lm_cmd += " --model_config prenorm-roberta-base"
    lm_cmd += " --cls_tokens 1"
    lm_cmd += " --wandb_name drop_tricks_α0_sf1_10kW_@2_12_6@_1_1_50_0_m1"  #
    lm_cmd += " --dataset /home/ahemf/processed_datasets/all512"
    lm_cmd += " --log_every_steps 50 --num_workers 8 --save_every_steps 2000"
    # lm_cmd += " --wandb_dryrun"
    lm_cmd += " --init_method=tcp  --master_addr 0.0.0.0 --master_port 9998 --shuffle_dataset --accumulation_steps 1"
    lm_cmd += " --epochs 3 --lr 0.004 --batch_size 16 --lr_steps 1 --weight_decay 0.005"
    lm_cmd += " --warmup_steps 10000 --gradient_clipping 3.0 --teacher_warmup_steps 1000"
    lm_cmd += " --sentence_order_prediction_w 1.0 --generator_w 1.0 --discriminator_w 50.0 --dino_w 0.0"  # --attention_penalty_w 100.0
    lm_cmd += " --optimizer adamw"
    lm_cmd += " --no_autocast"
    lm_cmd += " --lm_layers 2 --electra_layers 6 --lm_layers_total 12 --electra_layers_total 12"
    lm_cmd += " --drop_unused_layers"
    # lm_cmd += " --consecutive_layers"
    # lm_cmd += " --enable_layer_normalizers"
    lm_cmd += " --sampling_alpha 0.0"
    lm_cmd += " --start_from_proba 1.0"
    # lm_cmd += " --keep_last_layer"
    # lm_cmd += " --exclude_layers"
    lm_cmd += " --lm_temperature 1.0"
    lm_cmd += " --beta_1 0.9 --beta_2 0.98"
    # lm_cmd += " --move_unused_to_cpu"
    # lm_cmd += " --pretrained_model roberta-base"
    # lm_cmd += " --pretrained_model /home/ahemf/model_save_dir/mtt.pth"
    lm_cmd += " > output.log 2>&1 & disown"

    # > my.log 2>&1 &
    # cmd0 = "kill -2 $(ps aux | grep train_lm_distributed.py | grep -v grep | awk \'{print $2}\')"
    # cmd1 = "kill -2 $(ps aux | grep multiprocessing | grep -v grep | awk \'{print $2}\')"
    cmd0 = "pkill -9 -f 'train_'"
    cmd1 = "pkill -9 -f 'multiprocessing'"
    cmd_kill_nvidia = "nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9"
    cmd2 = "rm ~/torch_distributed_init/file-9999"
    cmd3 = cmd_dir + " && git fetch --all && git reset --hard origin/master && git pull"
    clear_log = cmd_dir + " && rm output.log"
    if args["kill"]:
        run_command_v2(hosts, cmd1)
        run_command_v2(hosts, cmd0)
        # run_command_v2(h1, cmd2)
        run_command_v2(hosts, clear_log)
        time.sleep(10)
    if args["ggl"]:
        run_command_v2(hosts, cmd3)
    if args["start"]:
        cmd4 = cmd_dir + " && " + main_cmd
        if "tcp" in main_cmd:
            ip_address_cmd = cmd_dir + " && " + "/usr/sbin/ifconfig eth0 | grep inet | cut -d: -f2"
            ipaddr = one_run(hosts[0], ip_address_cmd)["stdout"].strip().split()[1]
            part0, part1 = cmd4.split("--master_addr")
            part1 = part1.strip().split()[1:]
            part1[1] = "9999"
            part1 = ["--master_addr" + " " + ipaddr] + part1
            part1 = " ".join(part1)
            cmd4 = part0 + " " + part1

        run_command_v2(hosts, cmd4, list(zip([len(hosts)] * len(hosts), list(map(str, list(range(len(hosts))))))), args["ds"])

    if args["vision"]:
        cmd4 = cmd_dir + " && " + vision_cmd
        if "tcp" in main_cmd:
            ip_address_cmd = cmd_dir + " && " + "/usr/sbin/ifconfig eth0 | grep inet | cut -d: -f2"
            ipaddr = one_run(hosts[0], ip_address_cmd)["stdout"].strip().split()[1]
            part0, part1 = cmd4.split("--master_addr")
            part1 = part1.strip().split()[1:]
            part1[1] = "9999"
            part1 = ["--master_addr" + " " + ipaddr] + part1
            part1 = " ".join(part1)
            cmd4 = part0 + " " + part1

        run_command_v2(hosts, cmd4, list(zip([len(hosts)] * len(hosts), list(map(str, list(range(len(hosts))))))), args["ds"])

    if args["lm"]:
        cmd4 = cmd_dir + " && " + lm_cmd
        if "tcp" in main_cmd:
            ip_address_cmd = "/usr/sbin/ifconfig eth0 | grep inet | cut -d: -f2"
            addr = one_run(hosts[0], ip_address_cmd)["stdout"]
            print(addr)
            ipaddr = addr.strip().split()[1]
            part0, part1 = cmd4.split("--master_addr")
            part1 = part1.strip().split()[1:]
            part1[1] = "9999"
            part1 = ["--master_addr" + " " + ipaddr] + part1
            part1 = " ".join(part1)
            cmd4 = part0 + " " + part1

        run_command_v2(hosts, cmd4, list(zip([len(hosts)] * len(hosts), list(map(str, list(range(len(hosts))))))), args["ds"])

    if args["tail"]:
        tail_cmd = cmd_dir + " && tail -n %s output.log" % args["ntail"]
        run_command_v2(hosts, tail_cmd)
    if args["gpustat"]:
        gpustat_cmd = cmd_dir + " && gpustat"
        run_command_v2(hosts, gpustat_cmd)
    if args["custom"] is not None:
        custom_cmd = cmd_dir + " && " + args["custom"]
        # python run_on_hosts.py --hosts_file hosts-medium.txt --custom 'source ~/.zshrc && mkdir processed_datasets' --nodes 0:32
        custom_cmd = args["custom"]
        run_command_v2(hosts, custom_cmd)
    if args["scp"] is not None:
        # python run_on_hosts.py --hosts_file hosts-medium.txt --scp "scp -rC ./setup-2.sh ahemf@%s:/home/ahemf" --nodes 0:32
        # python run_on_hosts.py --hosts_file hosts-medium.txt --scp "scp -rC ./setup-3.sh ahemf@%s:/home/ahemf" --nodes 0:32
        # python run_on_hosts.py --hosts_file hosts-medium.txt --custom 'chmod 777 setup-2.sh && ./setup-2.sh' --nodes 0:32
        # python run_on_hosts.py --hosts_file hosts-medium.txt --custom 'echo $SHELL && pwd && source ~/.zshrc && which python' --nodes 0:32
        # python run_on_hosts.py --hosts_file hosts-medium.txt --custom 'ls -ltrah setup-3.sh' --nodes 0:32
        # python run_on_hosts.py --hosts_file hosts-medium.txt --custom 'source ~/.zshrc && chmod 777 setup-3.sh && ./setup-3.sh' --nodes 0:32
        # python run_on_hosts.py --hosts_file hosts-medium.txt --custom 'source ~/.zshrc && python -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)"' --nodes 0:32
        # python run_on_hosts.py --hosts_file hosts-medium.txt --custom 'cat /proc/mounts | grep torch_distributed_init' --nodes 1:32
        # python run_on_hosts.py --hosts_file hosts-medium.txt --scp "ssh dev-dsk-ahemf-datasets-i3-8x-623502bc.us-west-2.amazon.com 'scp -qrC -o StrictHostKeyChecking=no /local/processed_datasets/train_fastformer_resampled_50M ahemf@%s:/home/ahemf/processed_datasets >> output.log 2>&1 & disown'" --nodes 0:32
        # python run_on_hosts.py --hosts_file hosts-small.txt --custom 'source ~/.zshrc && pip install torch torchvision torchaudio && python -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)"' --nodes 0:16
        # python run_on_hosts.py --hosts_file hosts-medium.txt --custom 'cat /proc/mounts | grep torch_distributed_init' --nodes 10:16
        # python run_on_hosts.py --hosts_file hosts-medium.txt --custom 'source ~/.zshrc && pip install -U torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html > /dev/null && python -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)"' --nodes 0:32

        # python run_on_hosts.py --hosts_file hosts-medium.txt --custom 'source ~/.zshrc && python -c "import wandb; wandb.login(\"never\", \"7fcf597a1a07dcb2e98622a96838a7eb41b1245d\")"' --nodes 0:32

        # python run_on_hosts.py --hosts_file hosts-medium.txt --custom 'source ~/.zshrc && pip install -U protobuf wandb' --nodes 0:32
        # python run_on_hosts.py --hosts_file hosts-small.txt --custom 'source ~/.zshrc && wget https://raw.githubusercontent.com/pytorch/pytorch/master/torch/utils/collect_env.py && python collect_env.py' --nodes 0:16
        # python run_on_hosts.py --hosts_file hosts-medium.txt --custom 'source ~/.zshrc && pip uninstall torch -y && pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html' --nodes 0:32
        run_command_v2(hosts, args["scp"])

# python run_on_hosts.py --hosts_file hosts-large.txt --custom 'sudo mkdir /home/ahemf && sudo chown -R ahemf /home/ahemf' --nodes 0:64
# python run_on_hosts.py --hosts_file hosts-large.txt --scp "scp -rC ./setup-1.sh ahemf@%s:/home/ahemf" --nodes 0:64
# python run_on_hosts.py --hosts_file hosts-medium.txt --custom 'chmod 777 setup-1.sh && ./setup-1.sh > /dev/null' --nodes 0:32

# python run_on_hosts.py --hosts_file hosts-medium.txt --custom 'chmod 777 setup-2.sh && ./setup-2.sh > /dev/null' --nodes 0:32

# python run_on_hosts.py --hosts_file hosts-medium.txt --custom 'source ~/.zshrc && mkdir /home/ahemf/torch_distributed_init && sudo mount -t nfs -o nolock dev-dsk-ahemf-p3-us-east-1a-1-f698d178.us-east-1.amazon.com:/home/ahemf/torch_distributed_init /home/ahemf/torch_distributed_init' --nodes 1:32

# python run_on_hosts.py --hosts_file hosts-small.txt --custom 'source ~/.zshrc && pip uninstall torch -y && pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html > /dev/null' --nodes 0:16


# python run_on_hosts.py --hosts_file hosts-large.txt --custom 'source ~/.zshrc && python -c "import wandb; wandb.login(\"never\", \"7fcf597a1a07dcb2e98622a96838a7eb41b1245d\")"' --nodes 0:64


