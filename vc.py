import argparse
import numpy as np
from scipy.special import comb

parser = argparse.ArgumentParser(description='Calculate dvc from q')

parser.add_argument('q', help='q for calculate', type=int)
args = parser.parse_args()

def get_dvc(q, Nmax = 10000):
    mH = np.zeros(Nmax, dtype=np.int32)
    mH[0] = 1
    mH[1] = 2
    dvc = 1
    for i in range(2, Nmax + 1):
        mH[i] = 2 * mH[i - 1]
        if (i - 1 >= q): mH[i] -= comb(i - 1, q)
        print("mH({}) = {}".format(i, mH[i]))
        if (mH[i] == (2**i)):
            dvc = i
        else: 
            break
    return dvc

dvc = get_dvc(args.q)
print(dvc)
