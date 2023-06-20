# !/bin/bash

# a script to setup everything in shell 
sudo apt-get update
sudo apt install build-essential
sudo apt install git net-tools gcc

# 1. install riscv-toolchain
sudo apt-get install autoconf automake autotools-dev curl python3 python3-pip libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev ninja-build git cmake libglib2.0-dev
git clone https://github.com/riscv/riscv-gnu-toolchain
cd riscv-gnu-toolchain
git submodule update --init --recursive

sudo mkdir /opt/riscv32i
sudo chown $USER /opt/riscv32i
sudo mkdir /opt/riscv32im
sudo chown $USER /opt/riscv32im
sudo mkdir /opt/riscv32imc
sudo chown $USER /opt/riscv32imc

mkdir build; cd build
../configure --with-arch=rv32i --prefix=/opt/riscv32i
make -j$(nproc)
../configure --with-arch=rv32im --prefix=/opt/riscv32im
make -j$(nproc)
../configure --with-arch=rv32imc --prefix=/opt/riscv32imc
make -j$(nproc)

echo 'export PATH="/opt/riscv32i/bin:$PATH"' >> ~/.bashrc
echo 'export PATH="/opt/riscv32im/bin:$PATH"' >> ~/.bashrc
echo 'export PATH="/opt/riscv32imc/bin:$PATH"' >> ~/.bashrc