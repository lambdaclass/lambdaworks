# Ring-LWE Encryption

Educational implementation of Ring-LWE public-key encryption over the polynomial ring $R_q = \mathbb{Z}_q[X]/(X^N + 1)$, using lambdaworks' `PolynomialRingElement` with Dilithium parameters ($q = 8380417$, $N = 256$).

## Disclaimer

This implementation is not cryptographically secure due to non-constant time operations and other considerations. It is intended as an educational example showing the efficiency gains of Ring-LWE over plain LWE.

## What is Ring-LWE?

**Ring-LWE** replaces LWE's random matrix $A \in \mathbb{Z}_q^{m \times n}$ with a single polynomial $a \in R_q = \mathbb{Z}_q[X]/(X^N + 1)$. Multiplication by $a$ in the ring acts like matrix-vector multiplication but with much less data:

| | LWE | Ring-LWE |
|---|---|---|
| Public parameter | Matrix $A$: $m \times n$ scalars | Polynomial $a$: $N$ coefficients |
| Public key size | $m \times n + m$ elements | $2N$ elements |
| Encryption cost | Matrix-vector multiply $O(mn)$ | Ring multiply $O(N \log N)$ via NTT |

For Dilithium parameters ($N = 256$), a Ring-LWE public key is just 512 field elements, while an equivalent LWE system would need thousands.

## Mathematical Background

The ring $R_q = \mathbb{Z}_q[X]/(X^N + 1)$ consists of polynomials of degree $< N$ with coefficients modulo $q$. The key property is $X^N \equiv -1$: multiplying polynomials wraps around with a sign flip, creating a negacyclic convolution.

### Key Generation

1. Sample random $a \in R_q$ (all $N$ coefficients uniform in $[0, q)$)
2. Sample small secret $s \in R_q$ with $|s_i| \leq \eta$
3. Sample small error $e \in R_q$ with $|e_i| \leq \eta$
4. Compute $b = a \cdot s + e \in R_q$
5. Public key: $(a, b)$, Secret key: $s$

### Encryption (single bit $\mu \in \{0, 1\}$)

1. Sample small $r, e_1, e_2 \in R_q$
2. Compute $u = a \cdot r + e_1$
3. Compute $v = b \cdot r + e_2 + \mu \cdot \lfloor q/2 \rfloor$
4. Ciphertext: $(u, v)$

### Decryption

1. Compute $d = v - s \cdot u$
2. Check the constant coefficient: if closer to $\lfloor q/2 \rfloor$ than to $0$, output 1; otherwise 0

**Why it works**: $d = v - s \cdot u = (b \cdot r + e_2 + \mu \lfloor q/2 \rfloor) - s \cdot (a \cdot r + e_1) = e \cdot r - s \cdot e_1 + e_2 + \mu \lfloor q/2 \rfloor$. All the error terms are products/sums of small polynomials, so they remain small, and $d \approx \mu \lfloor q/2 \rfloor$.

## Example

```rust
use ring_lwe::{keygen, encrypt, decrypt};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

let mut rng = ChaCha20Rng::seed_from_u64(12345);

// Generate keys with error bound 2
let (pk, sk) = keygen(&mut rng, 2);

// Encrypt and decrypt
let ct = encrypt(&mut rng, &pk, 1);
assert_eq!(decrypt(&sk, &ct), 1);
```

## From Ring-LWE to Kyber and Dilithium

Ring-LWE is one step away from the actual NIST standards:

- **Kyber (ML-KEM)** uses **Module-LWE**: a $k \times k$ matrix of ring elements instead of a single $a$. This gives a flexible security/efficiency tradeoff by adjusting $k$ (2, 3, or 4).
- **Dilithium (ML-DSA)** uses **Module-LWE/SIS** for digital signatures with a Fiat-Shamir-with-aborts framework.

The ring $R_q = \mathbb{Z}_q[X]/(X^{256} + 1)$ and the NTT-based multiplication used here are exactly the same as in the real standards.

## References

- [Lyubashevsky, V., Peikert, C., Regev, O. "On Ideal Lattices and Learning with Errors over Rings" (2010)](https://doi.org/10.1007/978-3-642-13190-5_1)
- [FIPS 203: Module-Lattice-Based Key-Encapsulation Mechanism (ML-KEM)](https://csrc.nist.gov/pubs/fips/203/final)
- [FIPS 204: Module-Lattice-Based Digital Signature Standard (ML-DSA)](https://csrc.nist.gov/pubs/fips/204/final)
