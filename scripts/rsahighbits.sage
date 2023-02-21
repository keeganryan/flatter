#!/usr/bin/env sage

import argparse
import sys

BITLENGTH = 2048

def print_fplll_format(M):
    m, n = M.dimensions()
    s = "["
    for i in range(m):
        s += "["
        for j in range(n):
            s += str(M[i,j])
            if j < n - 1:
                s += " "
        s += "]"
        print(s)
        s = ""
    print("]")

def read_fplll_format():
    rows = []
    for line in sys.stdin:
        line = line.lstrip("[").rstrip("\n").rstrip("]")
        if len(line) == 0:
            break

        row = [int(x) for x in line.split(" ") if len(x) > 0 and x != "]"]
        rows += [row]
    m = len(rows)
    n = len(rows[0])
    for row in rows:
        assert len(row) == n

    L = Matrix(ZZ, m, n)
    for i in range(m):
        for j in range(n):
            L[i,j] = rows[i][j]
    return L
        

def gen_rsakey():
    prime_len = BITLENGTH // 2
    upper_bound = 2**prime_len
    # clamp upper bits to be 0b11
    lower_bound = (2**(prime_len - 1)) + (2**(prime_len - 2))
    p = random_prime(upper_bound, lower_bound)
    q = random_prime(upper_bound, lower_bound)
    N = p*q

    return N, (p, q)
    
def generate_problem_instance(unknown_bits):
    # First, generate RSA key
    N, (p, q) = gen_rsakey()

    p_msb = p - (p % (2**unknown_bits))

    # Secret information (p, q)
    # Public information (N, p_msb, unknown_bits)
    return (p, q), (N, p_msb, unknown_bits)

def get_kt(unknown_bits):
    assert BITLENGTH == 2048
    # Lookup table for optimal Coppersmith parameters
    kt_lut = [
        [340, (1, 2)], [408, (2, 3)], [437, (3, 4)], [454, (4, 5)],
        [464, (5, 6)], [471, (6, 7)], [476, (7, 8)], [480, (8, 9)],
        [484, (9, 11)], [486, (10, 11)], [488, (11, 12)], [490, (12, 13)],
        [492, (13, 15)], [493, (14, 15)], [494, (15, 16)], [495, (16, 17)],
        [496, (17, 18)], [497, (18, 19)], [498, (20, 21)], [499, (21, 22)],
        [500, (23, 24)], [501, (26, 27)], [502, (29, 30)], [503, (32, 33)],
        [504, (37, 38)], [505, (43, 44)], [506, (52, 53)], [507, (65, 66)],
        [508, (87, 88)], [509, (132, 133)], [510, (271, 273)],
    ]
    for allowed_unknown_bits, (k, t) in kt_lut:
        if allowed_unknown_bits >= unknown_bits + 1:
            return k, t
    raise "Not allowed"

def construct_lattice(prob, answer=None):
    N, p_msb, unknown_bits = prob
    k, t = get_kt(unknown_bits)

    PR = PolynomialRing(ZZ, 'x')
    x = PR.gens()[0]
    f = p_msb + x

    if answer is not None:
        # Check that f has a root r mod p
        p, q = answer
        r = p - p_msb
        assert f(r) % p == 0

    # Build table of powers of f and N
    f_powers = [f**0]
    N_powers = [N**0]
    for i in range(1, k + 1):
        f_powers += [f_powers[-1] * f]
        N_powers += [N_powers[-1] * N]

    aux_polys = []
    for i in range(k):
        g_i = N_powers[k - i] * f_powers[i]
        aux_polys += [g_i]

    for i in range(t):
        g_i = x**i * f_powers[k]
        aux_polys += [g_i]

    if answer is not None:
        # Check that all auxiliary polynomials are 0 mod p**k
        pk = p**k
        for g_i in aux_polys:
            assert g_i(r) % pk == 0

    dimension = t + k
    R = 2**unknown_bits
    L = Matrix(ZZ, dimension, dimension)
    for i, g_i in enumerate(aux_polys):
        scaled_g = g_i(R*x)

        coefs = scaled_g.list()
        for j, coef in enumerate(coefs):
            L[i,j] = coef

    return L

def solve_from_reduced_lattice(unknown_bits, L_red, answer=None):
    # Get single polynomial
    PR = PolynomialRing(ZZ, 'x')
    x = PR.gens()[0]

    R = 2**unknown_bits
    h = PR(L_red[0].list())(x / R)

    r = h.roots(ZZ)[0][0]
    assert h(r) == 0
    if abs(r) > R:
        raise "Recovery failed."
    lsbs = int(r)
    return lsbs

def attack_full(unknown_bits, seed=None):
    set_random_seed(seed)
    answer, problem = generate_problem_instance(unknown_bits)
    p, _ = answer
    print(f"Created problem with secret p=\n{hex(p)}")
    
    L = construct_lattice(problem, answer)

    L_red = L.LLL()

    N, p_msb, unknown_bits = problem
    p_lsb = solve_from_reduced_lattice(unknown_bits, L_red, answer=answer)

    print(f"LSBs are {hex(p_lsb)}")

    p_recovered = p_msb + p_lsb
    if p_recovered > 1 and p_recovered < N and N % p_recovered == 0:
        print("Recovery successful")
    else:
        print("Recovery failed")

def attack_generate_only(unknown_bits, seed=None):
    # Generate the lattice and print to stdout
    set_random_seed(seed)
    answer, problem = generate_problem_instance(unknown_bits)
    #p, _ = answer
    #print(f"Created problem with secret p=\n{hex(p)}")
    L = construct_lattice(problem, answer)
    print_fplll_format(L)

def attack_post_reduction_only(unknown_bits):
    L_red = read_fplll_format()
    p_lsb = solve_from_reduced_lattice(unknown_bits, L_red)
    print(f"Recovered LSBs are {hex(p_lsb)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unknown-bits", type=int, help="Number of unknown bits (0-510)", default=380)
    parser.add_argument("--step-1", action="store_true", help="Output unreduced lattice")
    parser.add_argument("--step-2", action="store_true", help="Solve from reduced lattice")
    parser.add_argument("--seed", type=int, help="RNG seed", default=0)
    
    args = parser.parse_args()

    unknown_bits = args.unknown_bits
    seed = args.seed

    if args.step_1 and (not args.step_2):
        attack_generate_only(unknown_bits, seed=seed)
    elif (not args.step_1) and args.step_2:
        attack_post_reduction_only(unknown_bits)
    else:
        attack_full(unknown_bits, seed=seed)

if __name__ == "__main__":
    main()
