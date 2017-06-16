#!/bin/bash

echo 'TENSORFLOW BENCHMARK'
echo 'Benchmark path: /tensorflow/vgg_benchmark.py'
echo 'Benchmark run inference in 3 iterations and 10 images in each batch'

python /tensorflow/vgg_benchmark.py

echo ''
echo 'CAFFE2 BENCHMARK'
echo 'Benchmark path: /caffe/vgg_benchmark.py'
echo 'Benchmark run inference in 3 iterations and 10 images in each batch'

python /caffe/vgg_benchmark.py
