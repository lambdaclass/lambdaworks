# Pohlig-Hellman Algorithm Implementation

This implementation demonstrates the Pohlig-Hellman algorithm for solving the discrete logarithm problem on elliptic curves. The algorithm is particularly efficient when the order of the group has small prime factors.

## Discrete Logarithm Problem

Given two points $P$ and $Q$ on an elliptic curve, where $Q = kP$ for some unknown integer $k$, the discrete logarithm problem consists of finding the value of $k$.

## Mathematical Foundations

### Group Structure

Let $G$ be a finite cyclic group of order $n$ with generator $g$. The Pohlig-Hellman algorithm exploits the structure of $G$ when $n$ has small prime factors. If $n$ can be factorized as:

$$n = \prod_{i=1}^{k} p_i^{e_i}$$

where $p_i$ are distinct primes and $e_i$ are positive integers, then the algorithm can solve the discrete logarithm problem efficiently.

### Algorithm Steps

1. **Factorization**: Decompose the group order $n$ into its prime power factors
2. **Subgroup Resolution**: For each prime power $p_i^{e_i}$:
   - Compute $g_i = g^{n/p_i^{e_i}}$
   - Compute $h_i = h^{n/p_i^{e_i}}$
   - Solve $h_i = g_i^{x_i}$ in the subgroup of order $p_i^{e_i}$
3. **Chinese Remainder Theorem**: Combine the solutions $x_i$ to find $x$ modulo $n$

### Subgroup Resolution Method

For each prime factor $p^e$ of the order, the algorithm:

1. Works in a subgroup of order $p^e$
2. Calculates the logarithm digit by digit in base $p$ using the following formula:
   $$x = x_0 + x_1 p + x_2 p^2 + \dots + x_{e-1} p^{e-1}$$
3. For each digit $x_i$, solves a discrete logarithm problem in a group of order $p$

### Chinese Remainder Theorem

Once the subproblems are solved, the algorithm uses the Chinese Remainder Theorem to find a solution that satisfies all congruences:

$$x \equiv a_1 \pmod{m_1}$$
$$x \equiv a_2 \pmod{m_2}$$
$$\vdots$$
$$x \equiv a_k \pmod{m_k}$$

The solution is given by:
$$x = \sum_{i=1}^{k} a_i M_i N_i \pmod{M}$$
where $M = \prod_{i=1}^{k} m_i$, $M_i = M/m_i$, and $N_i$ is the modular inverse of $M_i$ modulo $m_i$.

### Advantages

- Significantly reduces computational complexity when the order has small factors
- Perfectly complements algorithms like Baby-Step-Giant-Step

## Implementation

The implementation includes:

1. **Prime Factorization**: Efficient factorization of the group order
2. **Subgroup Operations**: Working with subgroups of prime power order
3. **Baby-Step Giant-Step**: Solving discrete logarithms in small groups
4. **Chinese Remainder Theorem**: Combining solutions from different subgroups

## Usage Example

```rust
// Create the test instance
let p = TestCurve::generator();
let order = get_point_order(&p);
// Factorization of 18 is 2 * 3²
let factors = vec![(2, 1), (3, 2)];

// Point Q = kP for an unknown k
let test_k = 7u64;
let q = p.operate_with_self(test_k);

// Solve the discrete logarithm
let found_k = pohlig_hellman(&p, &q, order, &factors);

// Verify the result
let computed_point = p.operate_with_self(found_k.unwrap());
assert_eq!(computed_point, q);
```

