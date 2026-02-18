# LWE Encryption (Regev)

Educational implementation of the Learning With Errors (LWE) public-key encryption scheme, based on Oded Regev's 2005 construction.

## Disclaimer

This implementation is not cryptographically secure. It uses toy parameters (q = 97, small dimensions) for readability and must not be used in production. It is intended as an educational example.

## What is LWE?

The **Learning With Errors** problem asks: given a random matrix $A \in \mathbb{Z}_q^{m \times n}$ and a vector $b = As + e$, where $s$ is a secret vector and $e$ is a "small" error vector, find $s$.

Without the error $e$, this is just a system of linear equations — solvable by Gaussian elimination. The small noise makes it computationally hard, even for quantum computers. This hardness assumption is the foundation of most post-quantum cryptographic schemes.

## Mathematical Background

### Key Generation

1. Choose secret $s \in \mathbb{Z}_q^n$ uniformly at random
2. Choose random matrix $A \in \mathbb{Z}_q^{m \times n}$
3. Sample small error $e \in \mathbb{Z}_q^m$ with $|e_i| \leq \eta$
4. Compute $b = As + e$
5. Public key: $(A, b)$, Secret key: $s$

### Encryption (single bit $\mu \in \{0, 1\}$)

1. Choose random binary vector $r \in \{0, 1\}^m$ (subset selection)
2. Compute $u = A^T r \in \mathbb{Z}_q^n$
3. Compute $v = b^T r + \mu \cdot \lfloor q/2 \rfloor \in \mathbb{Z}_q$
4. Ciphertext: $(u, v)$

### Decryption

1. Compute $d = v - s^T u$
2. If $d$ is closer to $\lfloor q/2 \rfloor$ than to $0$, output 1; otherwise output 0

**Why it works**: $d = v - s^T u = b^T r + \mu \lfloor q/2 \rfloor - s^T A^T r = e^T r + \mu \lfloor q/2 \rfloor$. Since $e$ and $r$ are small, $e^T r$ is small, so $d \approx \mu \lfloor q/2 \rfloor$.

## Example

```rust
use lwe::{keygen, encrypt, decrypt};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

let mut rng = ChaCha20Rng::seed_from_u64(42);

// Generate keys: dimension n=4, samples m=8, error bound 1
let (pk, sk) = keygen(&mut rng, 4, 8, 1);

// Encrypt a bit
let ct = encrypt(&mut rng, &pk, 1);

// Decrypt
assert_eq!(decrypt(&sk, &ct), 1);
```

## From LWE to Post-Quantum Cryptography

LWE is the starting point for understanding modern lattice-based cryptography:

| Scheme | Key idea | Used in |
|--------|----------|---------|
| **LWE** | Matrix $A$, vector operations | Foundation |
| **Ring-LWE** | Replace matrix with polynomial ring element | Efficiency |
| **Module-LWE** | Matrix of ring elements | Kyber (ML-KEM) |
| **Module-SIS** | Short integer solution over modules | Dilithium (ML-DSA) |

Each step trades generality for efficiency while maintaining (believed) quantum resistance.

## References

- [Regev, O. "On Lattices, Learning with Errors, Random Linear Codes, and Cryptography" (2005)](https://doi.org/10.1145/1060590.1060603)
- [Peikert, C. "A Decade of Lattice Cryptography" (2016)](https://web.eecs.umich.edu/~cpeikert/pubs/lattice-survey.pdf)
- [Peikert, C. Lattices in Cryptography — Lecture Notes](https://web.eecs.umich.edu/~cpeikert/lic15/lec01.pdf)
