#!/usr/bin/env bash

exe=$1
file=$2
first_line=$(head -n 1 $file)
cmd=$(head -n 1 $file | sed "s/[#/]* RUN: \(.*\)/\1/;s/\%s/$file/g;s/mlir-toy/$exe/g")

expect_fail=0
if [[ $cmd = not\ * ]]; then
  cmd=$(echo $cmd | sed 's/not *//')
  expect_fail=1
fi

bash -c "$cmd"

if [[ $expect_fail -eq 1 ]]; then
  if [[ $? -eq 0 ]]; then
    exit 1
  fi
fi

exit $?
