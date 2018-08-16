import os
import re
import sys
import numpy as np

def gflops(m, k, n, t):
    """
    Calculate the GFlops for a matrix multiplication kernel.
    """
    operations = m*k*n*2
    return operations*1e-9/t

def main():
    algo = sys.argv[1]
    matrix_size = [384, 512, 640, 768, 896, 1024, 1152, 1408, 1664, 
            1920, 2176, 2432, 2816, 3200, 3584, 3968, 4096]
    print "With algo: %s" % algo
    for ms in matrix_size:
        m, k, n = ms, ms, ms
        os.system("nvprof ./%s.out %s %s %s 2> res.log" % (algo, m, k, n))

        with open("res.log", "r") as f:
            l = f.readlines()
            t = l[4].split()[3]
            if re.compile(r'(\d+)(\.|)(\d+)us').match(t):
                t = eval(t[:-2])*1e-6
            if re.compile(r'(\d+)(\.|)(\d+)ms').match(t):
                print t
                t = eval(t[:-2])*1e-3
            # if re.compile(r'(\d+)(\.|)(\d+)s').match(t):
            #     t = eval(t[:-1])
            print "Matrix size: %s\tKernel performance: %.2f GFlop/s" % (ms, gflops(m, k, n, t))

if __name__ == '__main__':
    main()
