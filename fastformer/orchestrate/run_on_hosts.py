# pwd
# kill -2 $(ps aux | grep multiprocessing | grep -v grep | awk '{print $2}')
# kill -2 $(ps aux | grep train_lm_distributed.py | grep -v grep | awk '{print $2}')
# rm ~/torch_distributed_init/file-9999
# cd /home/ahemf/mygit/fastformer/fastformer/training
# git pull
# python run_on_hosts.py --hosts_file hosts.txt --scp "ssh dev-dsk-ahemf-i3-16x-52ee4831.us-west-2.amazon.com 'scp -qrC -o StrictHostKeyChecking=no /local/datasets/asin-images/trainer.pth ahemf@%s:/home/ahemf/model_save_dir >> output.log 2>&1 & disown'" --nodes 0:8
# python run_on_hosts.py --hosts_file hosts.txt --custom 'source ~/.zshrc && python -c "import wandb; wandb.login(\"never\", \"7fcf597a1a07dcb2e98622a96838a7eb41b1245d\")"' --nodes 0:64
# python run_on_hosts.py --hosts_file hosts.txt --custom 'source ~/.zshrc && df -h && free -g' --nodes 0:64
# python run_on_hosts.py --hosts_file hosts.txt --kill --nodes 0:64
# python run_on_hosts.py --hosts_file hosts.txt --custom 'source ~/.zshrc && gpustat' --nodes 0:64

# python run_on_hosts.py --hosts_file hosts.txt --custom 'sudo sudo sh cuda_11.4.0_470.42.01_linux.run --silent --driver --toolkit' --nodes 0:64

# python run_on_hosts.py --hosts_file hosts.txt --scp "scp -o StrictHostKeyChecking=no -rC ./setup-1.sh ahemf@%s:/home/ahemf" --nodes 0:64
# python run_on_hosts.py --hosts_file hosts.txt --custom 'chmod 777 setup-1.sh && ./setup-1.sh > /dev/null 2>&1 & disown' --nodes 0:64

# python run_on_hosts.py --hosts_file hosts.txt --scp "scp -o StrictHostKeyChecking=no -rC ./setup-2.sh ahemf@%s:/home/ahemf" --nodes 0:64
# python run_on_hosts.py --hosts_file hosts.txt --zsh --custom 'chmod 777 setup-2.sh && ./setup-2.sh > /dev/null 2>&1 & disown' --nodes 0:64

# python run_on_hosts.py --hosts_file hosts.txt --scp "scp -o StrictHostKeyChecking=no -rC ./setup-3.sh ahemf@%s:/home/ahemf" --nodes 0:64
# python run_on_hosts.py --hosts_file hosts.txt --zsh --custom 'chmod 777 setup-3.sh && ./setup-3.sh > /dev/null 2>&1 & disown' --nodes 0:64
# --working_dir /home/ahemf/mygit/offsite-tuning
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


import sys
sys.path.append(os.path.dirname(os.getcwd()))
from utils import one_run, justify

def one_run(host, cmd, arg=None, dry_run=False):
    import subprocess
    import shlex
    import shutil
    if "scp" in cmd:
        cur_cmd = cmd % (host)
        cmd_str = shlex.split(cmd % (host))
    else:
        cur_cmd = (cmd % arg) if arg is not None else cmd
        cmd_str = shlex.split("ssh -o StrictHostKeyChecking=no %s '%s'" % (host, cur_cmd))
    if dry_run:
        return {"host": host, "cmd": cur_cmd, "stdout": "", "stderr": ""}
    s = subprocess.run(cmd_str, shell=False, capture_output=True, text=True)
    return {"host": host, "stdout": s.stdout, "stderr": s.stderr, "cmd": " ".join(cmd_str)}


disown = " > output.log 2>&1 & disown"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hosts_file', required=True, type=str,
                        help='Pretrained Model')
    parser.add_argument('-n', '--nodes', default=None,
                        type=str, metavar='N')

    parser.add_argument("--ggl", default=False, action="store_true",
                        help="Flag to do something")
    parser.add_argument("--dry_run", default=False, action="store_true",
                        help="Flag to do something")
    parser.add_argument("--kill", type=str, default=None,required=False,
                        help="Kill Flag")
    parser.add_argument("--gpustat", default=False, action="store_true",
                        help="Flag to do something")
    parser.add_argument("--disown", default=False, action="store_true",
                        help="Disown Process for continuous running")
    parser.add_argument("--zsh", default=False, action="store_true",
                        help="Source zsh")
    parser.add_argument('--custom', required=False, type=str, default=None,
                        help='Custom cmd {0} = node rank, {1} = master_ipaddr, {2} = node name, {3} = num hosts')
    parser.add_argument('--working_dir', required=False, type=str, default=None,
                        help='Select Working Dir')
    parser.add_argument('--git_branch', required=False, type=str, default="main",
                        help='git_branch')
    parser.add_argument('--reinstall', required=False, type=str, default=None,
                        help='reinstall pip library')
    
    parser.add_argument('--scp', required=False, type=str, default=None,
                        help='scp cmd in format `scp -rC <from> <uname>@%s:<to>`')

    args = parser.parse_args()
    return vars(args)

def run_command_v2(hosts, cmd, dry_run=False, master_ipaddr=None, substitute=True):
    import re
    import shutil
    args = [None] * len(hosts)

    tsize = shutil.get_terminal_size()[0]
    cmds = []
    if substitute:
        for nr, name in enumerate(hosts):
            cmds.append(cmd.format(nr, master_ipaddr if nr != 0 else "127.0.0.1", name, len(hosts), len(hosts)*8, 9999))
    print(tabulate(list(zip(hosts, cmds)), headers=["host", "command"]))
    with ThreadPoolExecutor(min(64, len(hosts))) as executor:  # ProcessPoolExecutor is quite slower
        ld = list(executor.map(one_run, hosts, cmds, args, [dry_run] * len(hosts)))
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


def get_master_ip_addr(hosts):
    # ip_address_cmd = "/usr/sbin/ifconfig eth0 | grep inet | cut -d: -f2 | awk '{ print $2 }'"
    ip_address_cmd = "/usr/sbin/ifconfig eth0 | grep inet | cut -d: -f2"
    ipaddr = one_run(hosts[0], ip_address_cmd)["stdout"].strip().split()[1]
    return ipaddr


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

    
    if args["kill"]:
        # python run_on_hosts.py --hosts_file hosts.txt --kill "run_clm" --working_dir /home/ahemf/mygit/offsite-tuning
        # cmd0 = "kill -2 $(ps aux | grep train_lm_distributed.py | grep -v grep | awk \'{print $2}\')"
        # cmd1 = "kill -2 $(ps aux | grep multiprocessing | grep -v grep | awk \'{print $2}\')"
    
        cmd0 = "pkill -9 -f '%s'" % (args["kill"])
        cmd1 = "pkill -9 -f 'multiprocessing'"
        cmd_kill_nvidia = "nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9"
    
        clear_log = "source ~/.zshrc && "
        if args["working_dir"]:
            clear_log += "cd %s && " % args["working_dir"]
        clear_log += " rm output.log"

        run_command_v2(hosts, cmd0)
        run_command_v2(hosts, cmd1)
        run_command_v2(hosts, cmd_kill_nvidia, substitute=False)
        run_command_v2(hosts, clear_log)
        time.sleep(10)
    if args["ggl"]:
        # python run_on_hosts.py --hosts_file hosts.txt --ggl --git_branch main --working_dir /home/ahemf/mygit/offsite-tuning --nodes 0:16
        assert args["working_dir"]
        custom_cmd = "source ~/.zshrc && export NCCL_SOCKET_IFNAME=eth,enp6s0 && export NCCL_DEBUG=INFO && export NCCL_IB_GID_INDEX=3 &&"
        custom_cmd += "cd %s && " % args["working_dir"]
        git = "git -c filter.lfs.smudge= -c filter.lfs.required=false"
        custom_cmd += f"{git} fetch --all && {git} reset --hard origin/{args['git_branch']} && {git} -c filter.lfs.smudge= -c filter.lfs.required=false pull && echo `{git} log -1`"
        # custom_cmd += ("echo `git log -1`")
        run_command_v2(hosts, custom_cmd, dry_run=args["dry_run"])

    if args["gpustat"]:
        gpustat_cmd = "source ~/.zshrc && gpustat"
        run_command_v2(hosts, gpustat_cmd)
    if args["custom"] is not None:
        # python run_on_hosts.py --hosts_file hosts.txt --working_dir /home/ahemf/mygit/offsite-tuning --gpustat  --custom 'pip uninstall -y torch && pip install torch' --nodes 0:64
        # python run_on_hosts.py --hosts_file hosts.txt --working_dir /home/ahemf/mygit/offsite-tuning --gpustat  --custom 'tail -n10 output.log' --nodes 0:64
        # python run_on_hosts.py --hosts_file hosts.txt --working_dir /home/ahemf/mygit/offsite-tuning  --custom 'chmod 777 scripts/distill_emulator/gpt2-xl-2-base-multinode-student-patch.sh' --nodes 0:64
        # python run_on_hosts.py --hosts_file hosts.txt --zsh --custom "scripts/distill_emulator/gpt2-xl-2-base-multinode-student-patch.sh {3} {0} {1}" --working_dir /home/ahemf/mygit/offsite-tuning --disown
        # python run_on_hosts.py --hosts_file hosts.txt --working_dir /home/ahemf/mygit/ --zsh --custom 'git clone https://github.com/faizanahemad/accelerate.git && cd accelerate && pip install -e .'
        # python run_on_hosts.py --hosts_file hosts.txt --working_dir /home/ahemf/mygit/offsite-tuning  --custom "sed -i.bak -e '83,96d' /home/ahemf/anaconda3/lib/python3.10/site-packages/multiprocess/dummy/__init__.py" --nodes 0:16
        
        if args["zsh"]:
            custom_cmd = "source ~/.zshrc && "
        else:
            custom_cmd = ""

        if args["working_dir"]:
            custom_cmd += "cd %s && " % args["working_dir"]
        custom_cmd += args["custom"]
        if args["disown"]:
            custom_cmd = custom_cmd + disown
        
        ipaddr = get_master_ip_addr(hosts)
        run_command_v2(hosts, custom_cmd, master_ipaddr=ipaddr, dry_run=args["dry_run"])

    if args["reinstall"]:
        # python run_on_hosts.py --hosts_file hosts.txt --reinstall "numpy torch huggingface_hub threading"
        # --ignore-installed
        custom_cmd = "source ~/.zshrc && "
        custom_cmd += ("pip uninstall -y %s && pip install --no-cache-dir -U %s" % (args["reinstall"], args["reinstall"]))
        if args["disown"]:
            custom_cmd = custom_cmd + disown
        run_command_v2(hosts, custom_cmd, dry_run=args["dry_run"])


    if args["scp"] is not None:
        # Midway Init
        # python run_on_hosts.py --hosts_file hosts-medium.txt --custom 'source ~/.zshrc && python -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)"' --nodes 0:32
        # python run_on_hosts.py --hosts_file hosts-medium.txt --scp "ssh dev-dsk-ahemf-datasets-i3-8x-623502bc.us-west-2.amazon.com 'scp -qrC -o StrictHostKeyChecking=no /local/processed_datasets/train_fastformer_resampled_50M ahemf@%s:/home/ahemf/processed_datasets >> output.log 2>&1 & disown'" --nodes 0:32
        # python run_on_hosts.py --hosts_file hosts-small.txt --custom 'source ~/.zshrc && wget https://raw.githubusercontent.com/pytorch/pytorch/master/torch/utils/collect_env.py && python collect_env.py' --nodes 0:16
        run_command_v2(hosts, args["scp"])
        # python run_on_hosts.py --hosts_file hosts.txt --scp "ssh dev-dsk-ahemf-2a-42037424.us-west-2.amazon.com 'scp -qrC -o StrictHostKeyChecking=no /home/ahemf/mygit/offsite-tuning/emulators/gpt2-xl/8_2_2/student.pt ahemf@%s:/home/ahemf/mygit/offsite-tuning/emulators/gpt2-xl/8_2_2/ >> output.log 2>&1 & disown'" --nodes 4:8