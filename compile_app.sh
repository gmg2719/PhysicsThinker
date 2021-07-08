#!/bin/sh
PWD_PATH=`pwd`

compile_none() {
    echo "************************************"
    echo "*                                  *"
    echo "*                                  *"
    echo "*      @Compile basic modules      *"
    echo "*                                  *"
    echo "*                                  *"
    echo "************************************"
    cd ${PWD_PATH}
    if [ -d "./build" ]; then
        cd build
        rm -rf *
        cd ${PWD_PATH}
    else
        mkdir build
    fi
}

compile_basic() {
    echo "************************************"
    echo "*                                  *"
    echo "*                                  *"
    echo "*      @Compile basic modules      *"
    echo "*                                  *"
    echo "*                                  *"
    echo "************************************"
    cd ${PWD_PATH}
    if [ -d "./build" ]; then
        cd build
        rm -rf *
        cd ${PWD_PATH}
    else
        mkdir build
    fi
    echo "Compile the src/ module into .o files ..."
    cd src/
    g++ -c -O3 -std=c++11 -I${PWD_PATH}/include/ *.cpp
    mv *.o ../build/.
    cd ${PWD_PATH}
    cd src/common/
    g++ -c -O3 -std=c++11 -I${PWD_PATH}/include/ *.cpp
    mv *.o ../../build/.
    echo "Compile common/ module done !"
    cd ${PWD_PATH}
    cd src/na/
    g++ -c -O3 -std=c++11 -I${PWD_PATH}/include/ *.cpp
    mv *.o ../../build/.
    echo "Compile na/ module done !"
    cd ${PWD_PATH}
    cd src/geometry/
    g++ -c -O3 -std=c++11 -I${PWD_PATH}/include/ *.cpp
    mv *.o ../../build/.
    echo "Compile geometry/ module done !"
    cd ${PWD_PATH}
}

#echo "Start to compile and run phy_app !"
compile_basic
# The first application for the 2D-MOC method in reactor core neutron transport
echo "compile openmoc-2d app ..."
cd phy_app/openmoc-2d
g++ -c -std=c++11 -I${PWD_PATH}/include/ *.cpp
mv *.o ../../build/.
cd ${PWD_PATH}/build/
g++ *.o -o openmoc2d.exe -lpthread -lrt
echo "run openmoc-2d app ..."
./openmoc2d.exe
# remove the main object file of the openmoc-2d application

# The other applications
compile_none
echo "compile CFD-1d app ..."
cd phy_app/cfd-1d
g++ -c -std=c++11 -I${PWD_PATH}/include/ *.cpp
mv *.o ../../build/.
cd ${PWD_PATH}/build/
g++ *.o -o cfd1d.exe -lpthread -lrt
echo "run cfd-1d app ..."
./cfd1d.exe
# remove the main object file of the cfd-1d application
