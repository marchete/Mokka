#!/bin/sh

if [ $# -ne 1 ]; then
    echo "$0: Takes one argument, the name of the AI to compile"
    exit 1
fi
AI_Name=$1 #Get the AI name you passed as a command line parameter
AI_Name="${AI_Name%.*}" #Remove extension in case you passed "V4.cpp" instead of "V4"

clang++-9 -std=c++17 -Iinclude -march=core-avx2 -mpopcnt -mfma -mavx2 -Ofast -funroll-loops -fomit-frame-pointer -finline *.hpp "$AI_Name.cpp" -lpthread
mv a.out "$AI_Name"
strip -S --strip-unneeded --remove-section=.note.gnu.gold-version --remove-section=.comment --remove-section=.note --remove-section=.note.gnu.build-id --remove-section=.note.ABI-tag "$AI_Name"
upx "$AI_Name" -9 --best --ultra-brute --no-backup --force

if [ ! -f ./mnist/train-images.idx3-ubyte ]; then
    echo "Warning, MINST not found! You need to download and decompress MNIST dataset on ./mnist folder!"
	echo "Use the command './DOWNLOAD_MNIST.sh'"
fi