CFLAGS ?= -O3 -march=armv8.2-a+simd+fp+fp16fml
CC=gcc

all: uk_exo 

uk_exo: uk_exo.o  main.o


uk_exo.c: generator.py
	exocc -o . --stem $(*F) $^


main.c: uk_exo.c 

.PHONY: clean
clean:
	$(RM) uk_exo uk_exo.* *.o 
	$(RM) -r __pycache__/
