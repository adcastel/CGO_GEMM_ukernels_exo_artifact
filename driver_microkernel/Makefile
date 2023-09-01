
##BLISHOME=$(HOME)/EXO_artifact/opt/blis
BLISFLAGS := -I${BLISHOME}/include/blis/ -L/${BLISHOME}/lib/ -lblis -lm
OMPFLAGS := 
CC := gcc
CFLAGS := -O3 -march=armv8.2-a+simd+fp+fp16fml 
CFLAGS += -DFP32

OBJECTS := gemm_blis_neon_fp32.o uk_exo.o 


all: driver 

driver: $(OBJECTS)
	$(CC) $(CFLAGS) main.c -o test_uk_blis $(OBJECTS) -DMMR=${MMR} -DNNR=${NNR} $(BLISFLAGS) $(OMPFLAGS) 


.c.o:
	        $(CC) $(CFLAGS)  -c $*.c


clean:
	rm *.o test_uk_blis


