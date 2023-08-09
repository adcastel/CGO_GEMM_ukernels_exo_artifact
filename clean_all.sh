#!/bin/bash

echo "Removing BLIS..."
rm -rf opt/blis
echo "... DONE"

echo "Cleaning EXO_ukr_generator ..."
cd EXO_ukr_generator
make clean
cd ..
echo "... DONE"

echo "Cleaning driver_microkernel ..."
cd driver_microkernel
make clean
cd ..
echo "... DONE"

echo "Cleaning gemm_blis_family ..."
cd gemm_blis_family
make clean
rm output/*
cd ..
echo "... DONE"
