#!/bin/bash -x
mkdir -p build
pushd build
cmake ..
make
