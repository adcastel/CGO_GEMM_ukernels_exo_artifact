# Welcome to the Exo Microkernel Generator Artifact

Please, read the REQUIREMENTS.txt file for the compilers that are used.
If you have a higher version, please change the makefiles for matching your compilers. Right now, the compilers used in the scripts are gcc-10 and python3.9.


# How to use this generator

# Executing step by step

1) Execute the build.sh file like:
   source build.sh  -> It is important to use source instead of ./ because it sets a environment variable
   This script checks the compilers and it they are installed it will build and install the Exo compiler and the BLIS library.
   The BLIS library will be installed inside the artifact folder and set automatically to the PATHS environment variables.

2) Run the microkernel_generator.sh script:
   ./microkernel_generator.sh
   This script will execute the generator presented in the paper and will generate the 8x12 microkernel as is described in the paper.
   For generating more microkernels, please modify the EXO_ukr_generator/generator.py file. There are some examples of different micro-kernel sizes.
   Once the microkernel is generated, it will execute a test of correctness.

3) (Optional) Execute the execute_ukernel_solo.sh script: 
   ./execute_ukernel_solo.sh
   This script performs the evaluation of the stand-alone microkernels (Figure 13).
   It will execute a driver with the different microkernel sizes.
   The code for the micro-kernels is already in the folder so there is not need to move files from one folder to another.

4) (Optional) Execute the execute_algorithm.sh file:
   ./execute_algorithm.sh
   This script will generate the results for figures 14, 15, and 17 of the paper.
   It will execute the combination of different gemms (squarish and rectangular).
   The code for the micro-kernels is already in the folder so there is not need to move files from one folder to another.

# Executing all in one

Just run the build_and_execute_all.sh
   ./build_and_execute_all.sh
   This file will go through the aforementioned steps :)

## Directory Structure

blis -> Directory containing the blis library.
build_and_execute_all.sh -> Script for building the software and execute the overall experimentation.
build.sh -> scrip that builds the software and sets the environment variables.
clean_all.sh -> script that cleans all the structure.
driver_microkernel -> Directory with the test for the microkernel solo experiment.
execute_algorithm.sh -> script for executing the combination of algorithm and microkernels.
execute_ukernel_solo.sh -> script for executing the microkernel experiment.
exo -> directory of exo software
EXO_ukr_generator -> Directory of the main part of the software, the microkernel generator
gemm_blis_family -> Directory of the GEMM algorithm
microkernel_generator.sh -> Script that generates the microkernel as it is explained in the paper
opt -> directory for the software installation
README.txt -> This file
REQUIREMENTS.txt -> Hardware and software requirements for the experimentation. 