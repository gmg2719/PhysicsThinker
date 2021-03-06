#!/bin/sh
# Cygwin uses Newlib, not GNU's libstdc++. When there's no -std=c++XX, current GCC defaults to 
# -std=gnu++11 (GCC 6 changes to gnu++14 by default).
# G++ defines _GNU_SOURCE on glibc targets, meaning that -std=c++NN is, contrary to the 
# documentation, not strict ISO C++.
echo "Compile the src/ module into .o files ..."
PWD_PATH=`pwd`
if [ -d "./build" ]; then
    cd build
    rm -rf *
    cd ${PWD_PATH}
else
    mkdir build
fi
cd src/
g++ -c -O3 -std=gnu++11 -mavx2 -mfma -I${PWD_PATH}/include/ *.cpp
mv *.o ../build/.
cd ${PWD_PATH}
cd src/common/
g++ -c -O3 -std=gnu++11 -mavx2 -mfma -I${PWD_PATH}/include/ *.cpp
mv *.o ../../build/.
cd ${PWD_PATH}
cd src/os/
g++ -c -O3 -std=gnu++11 -mavx2 -mfma -I${PWD_PATH}/include/ *.cpp
mv *.o ../../build/.
cd ${PWD_PATH}
cd src/na/
g++ -c -O3 -std=gnu++11 -mavx2 -mfma -I${PWD_PATH}/include/ *.cpp
mv *.o ../../build/.
cd ${PWD_PATH}
cd src/net/
g++ -c -O3 -std=gnu++11 -mavx2 -mfma -I${PWD_PATH}/include/ *.cpp
mv *.o ../../build/.
cd ${PWD_PATH}
cd src/signal/
g++ -c -O3 -std=gnu++11 -mavx2 -mfma -I${PWD_PATH}/include/ *.cpp
mv *.o ../../build/.
echo "Compile done !"
echo "Start to do test !"
cd ${PWD_PATH}
cd test/
# TEST_NAME="test_logger test_list test_thpool test_fft test_na test_positioning"
TEST_NAME="test_na"
for name in ${TEST_NAME}
do
    cd ${PWD_PATH}/test/
    g++ -c -O3 -std=gnu++11 -mavx2 -mfma -I${PWD_PATH}/include/ ${name}.cpp
    mv ${name}.o ../build/.
    cd ${PWD_PATH}/build/
    g++ *.o -o ${name} -lpthread -lrt
    ./${name}
    rm -f ${name}.o
    echo "Compile and run ${name} done !"
done
