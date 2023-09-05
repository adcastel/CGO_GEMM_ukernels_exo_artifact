#/bin/bash

echo "###################################"
echo "Checking required compilers"
echo "###################################"

gcc_min=10

if ! hash gcc; then
    echo "ERROR: gcc is not installed"
    return 1
fi

gcc_v=$(gcc --version | grep ^gcc | sed 's/^.* //g' | cut -f1 -d".")
if [ $gcc_v -lt $gcc_min ]; then
    echo "Update gcc version to 10.0.0 or higher"
    return 1
fi
echo "[*] gcc OK"

if ! hash python3; then
    echo "python 3 is not installed"
    return 1
fi

ver=$(python3 -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1\2/')
if [ "$ver" -lt "39" ]; then
    echo "This script requires python 3.9 or greater"
    return 1
fi
echo "[*] python3 OK"

if ! hash pip3; then
    echo "pip3 is not installed"
    return 1
fi
echo "[*] pip3 OK"

if ! hash gnuplot; then
    echo "gnuplot is not installed"
    return 1
fi
echo "[*] gnuplot OK"

echo "###################################"
echo "Building Exo. This may take a while..."
echo "###################################"
cd exo
python3 -m pip install -U setuptools wheel pytest attr
python3 -m pip install -r requirements.txt
python3 -m build
pip3 install dist/*.whl
cd ..

echo "###################################"
echo "Building Blis"
echo "###################################"
HERE=${PWD}
mkdir -p opt
cd blis
./configure --prefix=${HERE}/opt/blis  CC=gcc CXX=g++ auto #cortexa57
make -j 6 && make install
cd ..

export BLISHOME=${HERE}/opt/blis


