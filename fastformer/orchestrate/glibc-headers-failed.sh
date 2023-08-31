wget http://ftp.gnu.org/gnu/libc/glibc-2.27.tar.gz
tar -xvf glibc-2.27.tar.gz
cd glibc-2.27
mkdir build 
cd build
~/glibc/glibc-2.27/configure --prefix=$HOME/glibc/install
make -j8



wget https://rpmfind.net/linux/centos/8-stream/BaseOS/x86_64/os/Packages/glibc-headers-2.28-164.el8.x86_64.rpm
wget https://rpmfind.net/linux/centos/8-stream/BaseOS/x86_64/os/Packages/glibc-devel-2.28-164.el8.x86_64.rpm
wget https://rpmfind.net/linux/centos/8-stream/BaseOS/x86_64/os/Packages/glibc-all-langpacks-2.28-164.el8.x86_64.rpm
wget https://rpmfind.net/linux/centos/8-stream/BaseOS/x86_64/os/Packages/glibc-common-2.28-164.el8.x86_64.rpm
wget https://rpmfind.net/linux/centos/8-stream/BaseOS/x86_64/os/Packages/glibc-2.28-164.el8.x86_64.rpm
wget https://rpmfind.net/linux/centos/8-stream/BaseOS/x86_64/os/Packages/glibc-minimal-langpack-2.28-164.el8.x86_64.rpm
wget https://rpmfind.net/linux/centos/8-stream/BaseOS/x86_64/os/Packages/glibc-locale-source-2.28-164.el8.x86_64.rpm

sudo rpm -Uvh glibc-2.28-164.el8.x86_64.rpm glibc-common-2.28-164.el8.x86_64.rpm \
glibc-devel-2.28-164.el8.x86_64.rpm glibc-headers-2.28-164.el8.x86_64.rpm \
glibc-all-langpacks-2.28-164.el8.x86_64.rpm glibc-minimal-langpack-2.28-164.el8.x86_64.rpm \
glibc-locale-source-2.28-164.el8.x86_64.rpm

sudo yum-config-manager --add-repo=https://repo.extreme-ix.org/centos/8/extras/x86_64/os
# https://www.centos.org/download/mirrors/
# https://centoshelp.org/resources/repos/
sudo yum -y install http://rpms.remirepo.net/enterprise/remi-release-7.rpm

# https://ngelinux.com/how-to-install-glibc-package-version-2-28-on-rhel-7-unofficially-for-testing-purposes/
# https://serverfault.com/questions/894625/safely-upgrade-glibc-on-centos-7

sudo yum localinstall glibc-2.28-164.el8.x86_64.rpm glibc-common-2.28-164.el8.x86_64.rpm \
glibc-devel-2.28-164.el8.x86_64.rpm glibc-headers-2.28-164.el8.x86_64.rpm \
glibc-all-langpacks-2.28-164.el8.x86_64.rpm glibc-minimal-langpack-2.28-164.el8.x86_64.rpm \
glibc-locale-source-2.28-164.el8.x86_64.rpm
