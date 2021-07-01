#!/bin/sh
echo "Compile the src/ module into .o files ..."
PWD_PATH=`pwd`
if [ -d "./build" ]; then
    cd build
    rm -f *
    cd ${PWD_PATH}
else
    mkdir build
fi
cd src/
g++ -c -O3 -std=c++11 -I${PWD_PATH}/include/ *.cpp
mv *.o ../build/.
cd ${PWD_PATH}
cd src/common/
g++ -c -O3 -std=c++11 -I${PWD_PATH}/include/ *.cpp
mv *.o ../../build/.
echo "Compile common/ module done !"
cd ${PWD_PATH}
cd src/geometry/
g++ -c -O3 -std=c++11 -I${PWD_PATH}/include/ *.cpp
mv *.o ../../build/.
echo "Compile geometry/ module done !"
echo "Start to compile and run phy_app !"
cd ${PWD_PATH}
# The first application for the 2D-MOC method in reactor core neutron transport
cd phy_app/openmoc-2d
echo "compile openmoc-2d app ..."
g++ -c -std=c++11 -I${PWD_PATH}/include/ *.cpp
mv *.o ../../build/.
cd ${PWD_PATH}/build/
g++ *.o -o openmoc2d.exe -lpthread -lrt
echo "run openmoc-2d app ..."
./openmoc2d.exe
cd ${PWD_PATH}
# The other applications
