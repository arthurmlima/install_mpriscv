
CROSS=gcc
CFLAGS=

all: disncode.py main.c	 	
	$(CROSS) main.c
	./a.out
	python3 disncode.py filtered.txt
	rm -f a.out filtered.txt
clean:


.PHONY: clean
