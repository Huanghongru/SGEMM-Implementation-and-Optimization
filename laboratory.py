import os
import re
import sys
import numpy as np

def gflops(m, k, n, t):
    """
    Calculate the GFlops for a matrix multiplication kernel.
    """
    operations = eval(m)*eval(k)*eval(n)*2
    return operations*1e-9/t

def main():
    argv_ = sys.argv
    m, k, n = argv_[1], argv_[2], argv_[3]
    os.system("nvprof ./a.out %s %s %s 2> res.log" % (m, k, n))

    with open("res.log", "r") as f:
        l = f.readlines()
        t = l[4].split()[3]
        if re.compile(r'(\d+)(\.|)(\d+)us').match(t):
            t = eval(t[:-2])*1e-6
        if re.compile(r'(\d+)(\.|)(\d+)ms').match(t):
            t = eval(t[:-2])*1e-3
        if re.compile(r'(\d+)(\.|)(\d+)s').match(t):
            t = eval(t[:-1])
        print "Kernel performance: %.2f GFlop/s" % gflops(m, k, n, t)

if __name__ == '__main__':
    main()
