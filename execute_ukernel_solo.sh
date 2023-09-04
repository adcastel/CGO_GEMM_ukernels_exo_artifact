#/bin/bash

echo "#############################################"
echo "Executing driver for microkernel in solo mode"
echo "#############################################"

HERE=${PWD}
cd driver_microkernel
mkdir -p output
echo "NEON" >>  output/neon.dat
echo "BLIS" >>  output/blis.dat
echo "EXO" >>  output/exo.dat
echo "#UK " >> output/1st_col.dat 
export LD_LIBRARY_PATH=/${BLISHOME}/lib/:$LD_LIBRARY_PATH
for IMR in 8 4
do
	for INR in 12 8 4
	do
        echo "${IMR}x${INR} " >> output/1st_col.dat 
          make clean
          make MMR=${IMR} NNR=${INR}
	  echo "Starting with ${IMR}x${INR}"
          ./test_uk_blis BLIS ${IMR} ${INR} 512 512 1 5 
	  tail -1 output/BLIS_${IMR}_${INR}.dat >>  output/blis.dat
          ./test_uk_blis NEON ${IMR} ${INR} 512 512 1 5 
	  tail -1 output/NEON_${IMR}_${INR}.dat >>  output/neon.dat
          ./test_uk_blis EXO ${IMR} ${INR} 512 512 1 5 
	  tail -1 output/EXO_${IMR}_${INR}.dat >>  output/exo.dat
	  echo "Ending with ${IMR}x${INR}"
        done
done
paste output/1st_col.dat output/neon.dat output/blis.dat output/exo.dat > output/ukernel_results.dat
rm output/1st_col.dat
rm output/blis.dat
rm output/neon.dat
rm output/exo.dat
rm output/NEON* output/BLIS* output/EXO*
cat  output/ukernel_results.dat
cd ..
