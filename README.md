# MPRISCV - Cana

## Setup 
Clone this repository and go to the main folder

    $ git clone this/repo ; cd this/repo


Load the bitstream of mpriscv 
    
    $ sudo fpgautil -b mpriscv_wrapper.bit

Install the python libraries: 
    
    $ sudo apt install python-opencv
    
    $ pip3 install scikit-learn scikit-image scipy numpy natsort matplotlib 

    $ pip install flask

We have installed the RISCV gcc toolchain in UbuntuZynq. You may choose to install it in you local machine. If you are stuburn enough, be advised that you will take more then 10 hours of build, 16 GB of storage and you must increase the system swap. 
    
    $ TODO: Figure it out by yourself for now 

You must now compile the firmware code for the riscv. Go to `mpriscv/rv` and run the make file

    $ make program

This will write an array with the instructions to program the riscvs.



Go to mpriscv directory 

    $ cd mpriscv

Compile as shared libraries the program in C/C++ which loads the mpriscv firmware and issues images transactions between the Arm-Host and mpriscv 
    
    $ gcc -fPIC -shared smp.c -o mpriscv.so

Pun python script 

    $ python3 main.py



