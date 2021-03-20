# sudo mkdir /home/ahemf
# sudo chown -R ahemf /home/ahemf

sudo yum -y --skip-broken install ncurses-compat-libs gcc git vim-enhanced emacs xterm gpm-devel.x86_64 git yum-utils amazon-midway-init toolbox xauth lapack lapack-devel atlas tree cmake gcc-c++ yum-config-manager coreutils
sudo yum-config-manager --enable extras
sudo yum makecache
sudo yum -y --skip-broken groupinstall 'Development Tools'
sudo yum -y --skip-broken install pciutils libglvnd libglvnd-devel rdma libibverbs
sudo yum -y install nfs-utils nfs-utils-lib portmap
sudo yum -y install llvm clang llvm-devel llvm-libs llvm-static
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
# sudo visudo # ahemf   ALL=(root) NOPASSWD:ALL,/bin/sh
# wget https://us.download.nvidia.com/tesla/450.102.04/NVIDIA-Linux-x86_64-450.102.04.run
# sudo chmod +x NVIDIA-Linux-x86_64*.run
# sudo sudo /bin/sh ./NVIDIA-Linux-x86_64*.run
# sudo reboot

# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile-advanced
# https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=CentOS&target_version=7&target_type=runfilelocal
# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=CentOS&target_version=7&target_type=runfilelocal
# https://docs.nvidia.com/cuda/archive/10.2/cuda-installation-guide-linux/

# wget https://developer.download.nvidia.com/compute/cuda/11.2.1/local_installers/cuda_11.2.1_460.32.03_linux.run
# chmod +x cuda_11.2.1_460.32.03_linux.run
# sudo sudo sh cuda_11.2.1_460.32.03_linux.run --silent --driver --toolkit

wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
chmod +x cuda_10.2.89_440.33.01_linux.run
sudo sudo sh cuda_10.2.89_440.33.01_linux.run --silent --driver --toolkit

wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/patches/1/cuda_10.2.1_linux.run
chmod +x cuda_10.2.1_linux.run
sudo sudo sh cuda_10.2.1_linux.run --silent --toolkit --driver

wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/patches/2/cuda_10.2.2_linux.run
chmod +x cuda_10.2.2_linux.run
sudo sudo sh cuda_10.2.2_linux.run --silent --toolkit --driver

sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
