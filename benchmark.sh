#!/bin/bash

#echo 'TENSORFLOW BENCHMARK'
#echo 'Benchmark path: /tensorflow/vgg_benchmark.py'
#python /tensorflow/vgg_benchmark.py

echo ''
echo 'CAFFE2 BENCHMARK'
echo 'Benchmark path: /caffe/vgg_benchmark.py'

python /caffe/vgg_benchmark.py

echo ''
echo 'SIRIN BENCHMARK'
cd /sirin/neuro && mix neuro.benchmark
