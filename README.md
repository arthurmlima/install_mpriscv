# MPRISCV - Sugar Cane

This repository contais the hardware platform configuration design: FPGA bitstream of a riscv based MPSoC _MPRISCV_ targeting  ZCU104 development board. It also includes a application for calculating angle rotation of sugar cane.

## Overview 

System flow should be something like: 

1. Configure the FPGA
2. Program the _MPRISCV_ 
3. Run Sugar Cane application


## Follow instructions 

The starting point I assume you have is an Ubuntu 22.04.2 OS for ZCU104. All the instructions are supposed to be executed in Zynq OS.

You should clone this repository and go to the main folder

    $ git clone this/repo ; cd this/repo

Load the bitstream of _MPRISCV_. Keep in mind that this bitstream upload will last only until its rebooted, after the reboot, the standard bsp .bit will be reloaded and the FPGA unconfigured.
    
    ~/reponame/$ sudo fpgautil -b mpriscv_wrapper.bit


Next we need to install the necessary python packages for the sugar cane app. 

What this app does under the hood is calling a dinamically linked library compile for C/C++ which access `/dev/mem`. This is relevant because you must run python app routine as sudo in order to meet permissions requirements and for some unknown odd reason to me, pip3 packages installed by users are not seen by sudo. 

I'm quite certain there is someway of add `$USER` to package lists shell commands, but for now, you should run the pip3 as sudo like: 

    ~$ pip3 install scikit-learn scikit-image scipy numpy natsort matplotlib 

OpenCV is another library used in this project 

    ~$ sudo apt install python-opencv
    
We have installed the RISCV gcc toolchain in UbuntuZynq. You may choose to install it in you local machine. If you are stuburn enough, be advised that you will take more then 10 hours of build, 16 GB of storage and you must increase the system swap. 
    
    $ TODO: Figure it out by yourself for now 

You must now compile the firmware code for the riscv. Go to `mpriscv/rv` and run the make file

    ~/reponame/mpriscv/rv/$ make program

This will write an array with the instructions to program the riscvs.

Go to mpriscv directory 

    ~/reponame/mpriscv/$ cd ..

Compile as shared libraries the program in C/C++ which loads the mpriscv firmware and issues images transactions between the Arm-Host and mpriscv 
    
    $ gcc -fPIC -shared smp.c -o mpriscv.so

Go to `core` folder for runing the sugar cane application

    $ cd ~/reponame/core

Run the application as sudo  

    $ sudo python3 main.py

The results should appear in `~/reponame/images/output_images/`




