# Pohlig-Hellman Attack Implementation

This implementation demonstrates the Pohlig-Hellman algorithm for solving the discrete logarithm problem on elliptic curves. The algorithm is practical when the group order has small prime factors.

This attack is significantly more efficient than attempting to solve the discrete logarithm problem using a brute-force search. In the group used for our implementation, the logarithm can be recovered in under one second with the Pohlig-Hellman attack, whereas a brute-force approach would take longer. This highlights the importance of working with groups of large prime order in cryptographic applications. For example, the group of the BLS12-381 elliptic curve over $\mathbb{F}_p$ has composite order $n$, and we need to ensure we are always working over a subgroup of large prime order $r$.

## Discrete Logarithm Problem

Given two points $g$ and $h$ on an elliptic curve, where $$h = g^x$$ for some unknown integer $x$, the discrete logarithm problem consists of finding the value of $x$.

## Mathematical Foundations

### Group Structure

Let $G$ be a finite cyclic group of order $n$ with generator $g$. Let's say we want to find $x$ such that $h = g^x$. The Pohlig-Hellman algorithm exploits the structure of $G$ when $n$ has small prime factors. In other words, if the factorization of $n$ is:

$$n = \prod_{i=1}^{k} p_i^{e_i} = p_1^{e_1} \ldots p_k^{e_k},$$

where $p_i$ are small distinct primes and $e_i$ are positive integers, then the algorithm can find $x$ efficiently.

### Algorithm Steps

1. **Factorization**: Decompose the group order $n$ into its prime power factors.
2. **Subgroup Resolution**: For each prime power $p_i^{e_i}$:
   - Compute $g_i = g^{n/p_i^{e_i}}$.
   - Compute $h_i = h^{n/p_i^{e_i}}$.
   - Solve $h_i = g_i^{x_i}$ in the subgroup of order $p_i^{e_i}$, using the Baby Step Giant Step algorithm. This will give us $x_i$ such that $x \equiv x_i \text{ mod } p_i^{e_i}$.
3. **Chinese Remainder Theorem**: Combine the solutions $x_i$ to find $x$ modulus $n$.

### Baby Step Giant Step Algorithm

Given a generator point $g$ of a subgroup of order $n$ and another point $h$ such that 
$$ h = g^x \text { mod } n$$
the algorithm finds $x$, reducing the complexity of $O(n)$ (using brute force) to $O(\sqrt{n})$. 

#### Algorithm Steps

1. Compute the step size $m = \lceil \sqrt{n} \rceil$.
2. Compute and store the list of points $\{g^0, g^1, \ldots, g^{m-1} \}$
3. We compute each element of another list $\{hg^{-0m}, hg^{-1m}, hg^{-2m}, \ldots, hg^{-m^2}\}$, and look for a coincedence between both lists.
4. If a match $g^j = hg^{-im}$ is found then the descrete log is $x = im + j$ because:
$$
\begin{aligned}
g^j &= hg^{-im} \\
g^{im + j} &= h
\end{aligned}
$$
5. If no match is found after $m$ iterations, then no solution exists within the given subgroup.

### Chinese Remainder Theorem

Once the subproblems are solved, the algorithm uses the [Chinese Remainder Theorem](https://en.wikipedia.org/wiki/Chinese_remainder_theorem) to find a solution that satisfies all congruences:

$$x \equiv x_1 \pmod{n_1}$$
$$x \equiv x_2 \pmod{n_2}$$
$$\vdots$$
$$x \equiv x_k \pmod{n_k}$$

The solution is given by:
$$x = \sum_{i=1}^{k} a_i M_i N_i \pmod{M}$$
where $N = \prod_{i=1}^{k} n_i$, $N_i = N/n_i$, and $M_i$ is the modular inverse of $N_i$ modulus $n_i$.

## Implementation

You'll find two files in this folder. The file `chinese_remainder_theorem` contains all the algorithms needed to compute the final step. 

In the other file, called `pohlig_hellman`, we define a group vulnerable to this attack and the algorithms that compute it.

We chose to use a subgroup $H$ of the BLS12-381 Elliptic Curve of order 
$$s = 96192362849 = 11 \cdot 10177 \cdot 859267.$$
This group is vulnerable to the attack because its factorization consists of small primes.

To find the generator $h$ of this group we took $g$ the generator of a bigger group of order $n$ and computed $$h = g^{\frac{n}{s}}.$$

### Usage Example

```rust
// Create the group
let group = PohligHellmanGroup::new();
let generator = &group.generator;
let order = &group.order;

// Define q = g^x.
let x = 7u64;
let q = generator.operate_with_self(x);

// Find x.
let x_found = group.pohlig_hellman_attack(&q).unwrap();

// Check the result.
assert_eq(generator.operate_with_self(x_found), q);
```
