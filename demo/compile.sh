#!/bin/sh
PWD_PATH=`pwd`
ROOT_PATH=$PWD_PATH/../

compile_none() {
    echo "************************************"
    echo "*                                  *"
    echo "*                                  *"
    echo "*      @Compile basic modules      *"
    echo "*                                  *"
    echo "*                                  *"
    echo "************************************"
    cd ${ROOT_PATH}/
    if [ -d "./build" ]; then
        cd build
        rm -rf *
        cd ${PWD_PATH}
    else
        mkdir build
    fi
}

compile_needed() {
    echo "************************************"
    echo "*                                  *"
    echo "*                                  *"
    echo "*      @Compile basic modules      *"
    echo "*                                  *"
    echo "*                                  *"
    echo "************************************"
    cd ${ROOT_PATH}
    if [ -d "./build" ]; then
        cd build
        rm -rf *
        cd ${ROOT_PATH}
    else
        mkdir build
    fi
    echo "Compile the src/ module into .o files ..."
    cd src/
    g++ -c -O3 -std=c++11 -I${ROOT_PATH}/include/ *.cpp
    mv *.o ../build/.
    cd ${ROOT_PATH}
    cd src/common/
    g++ -c -O3 -std=c++11 -I${ROOT_PATH}/include/ *.cpp
    mv *.o ../../build/.
    echo "Compile common/ module done !"
    cd ${PWD_PATH}
}

compile_needed
g++ -c -std=c++11 -I${ROOT_PATH}/include/ main.cpp
mv main.o ../build/.
cd ${ROOT_PATH}/build/
g++ *.o -o demo.exe -lpthread -lrt
mv demo.exe ${PWD_PATH}/.
echo "Compile demo.exe done !"
cd ${PWD_PATH}
echo "Run demo.exe !"
./demo.exe
cd ${ROOT_PATH}/build
rm -f *.o
cd ${PWD_PATH}
