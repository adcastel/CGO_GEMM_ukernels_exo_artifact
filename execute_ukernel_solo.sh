#/bin/bash

echo "#############################################"
echo "Executing driver for microkernel in solo mode"
echo "#############################################"

HERE=${PWD}
cd driver_microkernel
export LD_LIBRARY_PATH=/${BLISHOME}/lib/:$LD_LIBRARY_PATH
for IMR in 8 4
do
	for INR in 12 8 4
	do
          make clean
          make MMR=${IMR} NNR=${INR}
	  echo "Starting with ${IMR}x${INR}"
          ./test_uk_blis BLIS ${IMR} ${INR} 512 512 1 5 
          ./test_uk_blis NEON ${IMR} ${INR} 512 512 1 5 
          ./test_uk_blis EXO ${IMR} ${INR} 512 512 1 5 
	  echo "Ending with ${IMR}x${INR}"
        done
done
cd ..
