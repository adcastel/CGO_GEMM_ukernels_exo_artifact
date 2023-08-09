#/bin/bash

echo "###################################"
echo "Checking required compilers"
echo "###################################"


if ! hash gcc-10; then
    echo "ERROR: gcc is not installed"
    exit 1
fi

if ! hash python3.9; then
    echo "python3 is not installed"
    exit 1
fi

if ! hash pip3.9; then
    echo "pip3.9 is not installed"
    exit 1
fi

echo "###################################"
echo "Building Exo. This may take a while..."
echo "###################################"
cd exo
python3.9 -m pip install -U setuptools wheel pytest attr
python3.9 -m pip install -r requirements.txt
python3.9 -m build
pip3.9 install dist/*.whl
cd ..

echo "###################################"
echo "Building Blis"
echo "###################################"
HERE=${PWD}
mkdir -p opt
cd blis
./configure --prefix=${HERE}/opt/blis  CC=gcc-10 CXX=g++-10 auto #cortexa57
make -j 6 && make install
cd ..

export BLISHOME=${HERE}/opt/blis


