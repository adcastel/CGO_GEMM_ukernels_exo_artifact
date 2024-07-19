#/bin/bash

echo "#############################################"
echo "Executing GEMM algorithm"
echo "#############################################"

cd gemm_blis_family 
export LD_LIBRARY_PATH=/${BLISHOME}/lib/:$LD_LIBRARY_PATH
for RM in FAMILY_EXO
do
  make clean
  if [ $RM == "FAMILY_EXO" ]; then
      SM=ARMv8_EXO
  else
      SM=ARMv8
  fi
  make RUN_MODE=$RM SIMD_MODE=$SM 
  echo "Starting on ${RM} and ${SM} mode"
  ./run_gemm.sh cnn_models/square_small.dat output/square_small_${RM}.dat
  ./run_gemm.sh cnn_models/square.dat output/square_${RM}.dat
  ./run_gemm.sh cnn_models/resnet50.blis.dat output/rn50_${RM}.dat
  echo "End of ${RM} and ${SM} mode"
done
cd ..

echo "Test completed, check the output folder for the results"
