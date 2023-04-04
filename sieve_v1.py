import sys
import time
import random
import math
from functools import reduce
import numpy as np
from linalg import GF2

import cProfile
import io
import pstats
from pstats import SortKey


PROFILE = False


def profile(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE  # 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        if PROFILE:
            ps.print_stats()
            print(s.getvalue())
        return retval

    return wrapper



CONST_E = 2.7182818
CONST_LOG_10_E = math.log(CONST_E)
ln = lambda n: math.log(n) / CONST_LOG_10_E


# usual pi(n) approximation
def primes_less_than(n: int) -> int:
    return math.ceil( n / ln(n) )


# pull a few thousand small primes from a local list of them
def read_primes_from_file(filename):
    with open(filename, 'r') as file:
        primes_str = file.read()
        primes_list = [int(x) for x in primes_str.split()]
        primes_array = np.array(primes_list)
    return primes_array


CONST_SMALL_PRIMES = read_primes_from_file("primes.txt")

# convenient way to hold a number and its prime factorization
class factor_tree:
    def __init__(self, n):
        self.n = n
        self.factors = {}

    def add_factor(self, f):
        if self.factors.get(f):
            self.factors[f] += 1
        else:
            self.factors.update({f: 1})

    def is_b_smooth(self, b: int) -> bool:
        fs = [k for k in self.factors]
        if len(fs) == 0:
            return False
        else:
            return b >= max([k for k in self.factors])

    def to_exp_vec_g2(self, factor_base: list) -> np.array:
        return np.array([self.factors[p] % 2 if p in self.factors else 0 for p in factor_base])

    def __repr__(self):
        s = f"{self.n} = ["
        for k, v in self.factors.items():
            s += f"{k}^{v} * "
        s = s[:-3]
        s += "]"
        return s


def try_small_primes(n: int, small_primes: list) -> (int, int, list):
    f = []
    # first few k primes

    for p in small_primes:
        while n % p == 0:
            n //= p
            f.append(p)
        if n == 1:
            break

    return n, p, f

def rho(n):
    x = 2; y = 2; d = 1; c = 0;
    f = lambda x: (x**2 + 1) % n
    while d == 1:
        c += 1
        x = f(x); y = f(f(y))
        d = math.gcd(x-y, n)

    return d


def factor(n: int) -> factor_tree:
    f = factor_tree(n)
    n, last_p, f1 = try_small_primes(n, CONST_SMALL_PRIMES)
    for factor in f1:
        f.add_factor(factor)
    if n == 0:
        return factor_tree(0)
    return f


# returns b relations, i.e. b equations s.t. for (x, f):
# .. (sqrt(n) + x) ** 2 (mod n) == p_1 * p_2 * ...
def sieve(n: int, b: int) -> list:
    relations = []
    floor_sqrt_n = math.ceil( math.sqrt(n) )
    x = 0 
    print(f"using polynomial y(x) = ({floor_sqrt_n} + x)^2")
    print(f"searching for {primes_less_than(b)} relations...")
    while len(relations) < primes_less_than(b):
        y = pow( floor_sqrt_n + x , 2, n )
        f = factor(y)
        if f.n != 0 and f.is_b_smooth(b):
            # print(f"found relation for x={x}:  {f}")
            relations.append((x + floor_sqrt_n, f))
            # TODO: if we happened to find a fermat square, check gcd immediately
                # i.e. if every p in n has an even exponent, immediately check gcd
        
        x += 1
    
    print(f"found {len(relations)}...")
    return relations


# avert your eyes
def combine_relations(n: int, r: list) -> int:
    factor_base = reduce(np.union1d, [list(rel[1].factors.keys()) for rel in r])
    exp_matrix = np.array([rel[1].to_exp_vec_g2(factor_base) for rel in r])
    ys = np.array([rel[1].n for rel in r])
    xs = np.array([rel[0] for rel in r])

    # print(exp_matrix)

    g2 = GF2(exp_matrix)
    k = g2.co_ker()

    print(f"found basis of shape {k.shape} for left-null-space of exp. matrix")
    # print(ys)
    for basis in k:
        y_guess = int(math.sqrt(abs(np.product(np.array([yn if vn else 1 for yn, vn in zip(ys, basis)])))))
        # print(f"y_guess: {y_guess}")
        x_guess = np.prod(np.array([xn if vn else 1 for xn, vn in zip(xs, basis)]))
        # print(f"x_guess: {x_guess}")
        d = math.gcd(y_guess - x_guess, n)
        # print(f"gcd({y_guess} - {x_guess}, {n}) = {d}")
        if d != n and d != 1:
            return d

    return 0
        

@profile
def run(n):
    b = 20_000
    st0 = time.perf_counter()
    r = sieve(n, b)
    st1 = time.perf_counter()
    st = st1 - st0
    print(f"found relations in {st:.6f} seconds")
    dt0 = time.perf_counter()
    d = combine_relations(n, r)
    dt1 = time.perf_counter()
    dt = dt1 - dt0
    print(f"combined relations in {dt:.6f} seconds")

    if d != 0:
        print("found factorization!")
        f1 = d
        f2 = n / f1
        assert(f2 == int(f2))
        f2 = int(f2)
        print(f"{f1 * f2} = {f1} * {f2}")
    
    else:
        print("Failed to find a factor!")



if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'all':
            ns = [22504204457389, 50013360979327, 874290122567, 19912039333033, 16532565756937, 11607579409399, 
                  53721951108809, 2290151474879, 10488834374147, 60977625124747, 8011824956321, 277475436907,
                  41908143780523, 23280873878233, 1999879205591, 25709855395261, 7187004956569]
            for n in ns:
                run(n)
        else:
            n = int(sys.argv[1])
            run(n)
    else:
        run(4163679007)



