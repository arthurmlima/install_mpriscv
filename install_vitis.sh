


sudo dpkg --add-architecture i386
sudo dpkg-reconfigure dash
sudo apt install -y iproute2 gcc g++ net-tools libncurses5-dev zlib1g:i386 libssl-dev flex bison libselinux1 xterm autoconf libtool texinfo zlib1g-dev gcc-multilib build-essential screen pax gawk python3 python3-pexpect python3-pip python3-git python3-jinja2 xz-utils debianutils iputils-ping libegl1-mesa libsdl1.2-dev pylint3 cpio

cd ~
cp /mnt/d/XilinxData/XilinxTools/Xilinx_Unified_2020.1_0602_1208.tar.gz ~
cd ~
tar xvf Xilinx_Unified_2020.1_0602_1208.tar.gz
cd Xilinx_Unified_2020.1_0602_1208
sudo ./xsetup

echo 'source /tools/Xilinx/Vivado/2020.1/settings64.sh' >> ~/.bashrc
echo 'source /tools/Xilinx/Vitis/2020.1/settings64.sh'  >> ~/.bashrc

