sudo mkdir -p /home/ahemf
sudo chown -R ahemf /home/ahemf

sudo yum -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
sudo yum -y install yum-config-manager
sudo yum -y install yum-utils

sudo yum-config-manager --disable al-amazon al-kernel-ac al-main al-acc al-updates
sudo yum -y --disablerepo=al-acc --disablerepo=al-amazon --disablerepo=al-kernel-ac --disablerepo=al-main --disablerepo=al-updates install yum-config-manager
sudo yum -y --disablerepo=al-acc --disablerepo=al-amazon --disablerepo=al-kernel-ac --disablerepo=al-main --disablerepo=al-updates install yum-utils
sudo yum-config-manager --disable al-amazon al-kernel-ac al-main al-acc al-updates

sudo yum-config-manager --enable extras
sudo yum -y --skip-broken install ncurses-compat-libs gcc git vim-enhanced emacs xterm gpm-devel.x86_64 git yum-utils amazon-midway-init toolbox xauth lapack lapack-devel atlas tree cmake gcc-c++ coreutils
sudo yum makecache
sudo yum -y --skip-broken groupinstall 'Development Tools'
sudo yum -y --skip-broken install pciutils libglvnd libglvnd-devel rdma libibverbs
sudo yum -y install nfs-utils nfs-utils-lib portmap
sudo yum -y install llvm clang llvm-devel llvm-libs llvm-static
sudo yum-config-manager --add-repo http://dev-desktop-repos.amazon.com/Amazon-Dev-Desktop-GUI.repo
sudo yum -y install amazon-midway-init
# sudo yum -y remove nvidia-dkms nvidia
sudo yum -y install htop
sudo yum -y install nfs-utils nfs-utils-lib portmap


cd /home/ahemf

wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
chmod +x cuda_11.6.0_510.39.01_linux.run
sudo killall nvidia-smi
sudo pkill nvidia-smi

sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

cd /home/ahemf
cd ~
mkdir nvidia
sudo yum --showduplicates list kernel-headers | expand

sudo yum remove -y kernel-headers
sudo yum remove -y kernel-devel
sudo yum install -y kernel-headers-`uname -r`
sudo yum install -y kernel-devel-`uname -r`
sudo yum install -y systemtap systemtap-devel libtool lapack-devel llvm-devel llvm-test blas-devel gcc-gfortran clang gcc-c++ gcc glibc-devel glibc-headers screen

sudo sudo sh cuda_11.6.0_510.39.01_linux.run --silent --driver --toolkit --toolkitpath=$HOME/nvidia/toolkit --samples --samplespath=$HOME/nvidia/samples --tmpdir=$HOME
export PATH=$HOME/nvidia/toolkit/bin/:$PATH
export LD_LIBRARY_PATH=$HOME/nvidia/toolkit/lib64/:$LD_LIBRARY_PATH

sudo yum install https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
sudo yum clean all
sudo rm -rf /var/cache/yum

sudo yum install -y --downloadonly --downloaddir=$HOME datacenter-gpu-manager-2.1.7-1 
sudo yum localinstall datacenter-gpu-manager-2.1.7-1-x86_64.rpm
sudo systemctl --now enable nvidia-dcgm

# https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/
# https://pkgs.org/download/nvidia-fabric-manager
wget https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/nvidia-fabric-manager-510.39.01-1.x86_64.rpm
sudo yum localinstall nvidia-fabric-manager-510.39.01-1.x86_64.rpm
sudo systemctl --now enable nvidia-fabricmanager

# https://linuxconfig.org/how-to-disable-blacklist-nouveau-nvidia-driver-on-ubuntu-20-04-focal-fossa-linux
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# https://www.server-world.info/en/note?os=CentOS_7&p=nvidia&f=2