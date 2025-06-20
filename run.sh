#!/bin/bash

# Run all kernels
rm ./test/test_kernel*
echo -n "test_kernel:"
for((i=0;i<=7;i++))
do
    echo -n "${i}..."
    file_name="./test/test_kernel_${i}.txt"
	./sgemm ${i} >> ${file_name}
done

# Run a single kernel
# kernel_num=$1
# file_name="test_kernel_${kernel_num}.txt"
# ./sgemm ${kernel_num} | tee ./test/${file_name}