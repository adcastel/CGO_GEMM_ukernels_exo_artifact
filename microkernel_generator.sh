#/bin/bash

echo "###################################"
echo "Microkernel Generation"
echo "###################################"
cd EXO_ukr_generator
export PYTHONPATH=.:$PYTHONPATH
make clean; 
make
./uk_exo
cd ..
