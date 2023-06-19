sudo dpkg --add-architecture i386
sudo dpkg-reconfigure dash
sudo apt install -y iproute2 gcc g++ net-tools libncurses5-dev zlib1g:i386 libssl-dev flex bison libselinux1 xterm autoconf libtool texinfo zlib1g-dev gcc-multilib build-essential screen pax gawk python3 python3-pexpect python3-pip python3-git python3-jinja2 xz-utils debianutils iputils-ping libegl1-mesa libsdl1.2-dev pylint3 cpio

text="service tftp 
    {
    protocol = udp 
    port = 69 
    socket_type = dgram 
    wait = yes 
    user = nobody 
    server = /usr/sbin/in.tftpd 
    server_args = /tftpboot 
    disable = no
    }
    "

sudo echo -e "$text" > /etc/xinetd.d/tftp    

sudo mkdir /tftpboot
sudo chmod -R 777 /tftpboot
sudo chown -R nobody /tftpboot
sudo /etc/init.d/xinetd stop
sudo /etc/init.d/xinetd start

sudo apt update
sudo adduser $USER dialout



sudo mkdir -p /tools/Xilinx/PetaLinux/2020.1/
sudo chmod -R 755 /tools/Xilinx/PetaLinux/2020.1/
sudo chown -R $USER:$USER /tools/Xilinx/PetaLinux/2020.1/
cp /mnt/d/XilinxData/XilinxPetalinux/petalinux-v2020.1-final-installer.run ~
sudo chmod 777 ~/petalinux-v2020.1-final-installer.run
cd ~
./petalinux-v2020.1-final-installer.run --dir /tools/Xilinx/PetaLinux/2020.1/