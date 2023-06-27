
CROSS=riscv32-unknown-elf-
CFLAGS=

program: section.lds start.s ../firmware.c	 
	rm -f firmware.elf $(CWD)/../program.h dissassembly.dsm
	$(CROSS)gcc -mabi=ilp32 -march=rv32g -Wl,--build-id=none,-Bstatic,-T,section.lds,--strip-debug -ffreestanding -nostdlib -o firmware.elf ../firmware.c
	$(CROSS)objdump -s -j .data firmware.elf > diss.dsm	
	$(CROSS)objdump -d firmware.elf > dissassembly.dsm
	python3 disncode.py dissassembly.dsm ../program.h

clean:
	rm -f firmware.elf $(CWD)/../src/program.h dissassembly.dsm	


.PHONY: clean
