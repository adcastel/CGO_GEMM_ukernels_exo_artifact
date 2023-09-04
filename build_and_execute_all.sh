#/bin/bash

./clean_all.sh
source build.sh;
./microkernel_generator.sh 
./execute_ukernel_solo.sh 
./execute_algorithm.sh
./do_plots.sh
