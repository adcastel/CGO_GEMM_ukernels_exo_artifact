#!/bin/bash

#-------------------------------------------------------------
# ALPHA - BETA Fixed Values
#-------------------------------------------------------------
ALPHA=1.0   
BETA=0.0   
#-------------------------------------------------------------


#-------------------------------------------------------------
# Execution Minimum Time
#-------------------------------------------------------------
TIMIN=3.0 
#-------------------------------------------------------------


#-------------------------------------------------------------
# Enable (T) | Disable (F) Testing Mode
#-------------------------------------------------------------
TEST=N
#-------------------------------------------------------------


#-------------------------------------------------------------
# Enable (0) | Disable (1) Visual Mode
#-------------------------------------------------------------
VISUAL=0
#-------------------------------------------------------------

mkdir -p output

if $(echo $1 | grep -q "batch"); then 
	source $1
else
	source batch/.null.sh
fi


taskset -c 1 ./build/test_gemm.x "" "C" "C" "C" "N" "N" $ALPHA $BETA $MMIN $MMAX $MSTEP $NMIN $NMAX $NSTEP $KMIN $KMAX $KSTEP $VISUAL $TIMIN $TEST $1 $2

#./build/test_gemm.x "" $ORDERA $ORDERB $ORDERC $TRANSA $TRANSB $ALPHA $BETA $MMIN $MMAX $MSTEP $NMIN $NMAX $NSTEP $KMIN $KMAX $KSTEP $VISUAL $TIMIN $TEST $1 $2

