# README

Some source code about matrix multiplication implementation on CUDA.

## Device Properties

    --- General Information for device 0 ---
Name:  GeForce GTX 1080 Ti

Compute capability:  6.1

Clock rate:  1.68GHz

Device copy overlap: Enabled

Kernel execution timeout:  Disabled

    --- Memory Information for device 0 ---
Total global mem:   10.91G

Total constant mem:    64KB

Max mem pitch:    2147483647

Texture Alignment:    512

    --- MP Information for device 0 ---
Multiprocessor count:    28

Shared mem per blcok:    48KB

Registers per blcok:    65536

Threads in warp:    32

Max threads per block:    1024

Max thread dimensions:  (1024, 1024, 64)

Max grid dimensions: (2147483647, 65535, 65535)

## Miscellaneous

compile the file as follows:

```
nvcc *.cu --std=c++11
```

