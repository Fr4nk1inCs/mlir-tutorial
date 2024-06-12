#!/usr/bin/env bash

exe=$1
file=$2
first_line=$(head -n 1 $file)
cmd=$(head -n 1 $file | sed "s/# RUN: \(.*\)/\1/;s/\%s/$file/g;s/mlir-toy/$exe/g")

bash -c "$cmd"

exit $?
