# Lattice-based Cryptography

This module provides foundational building blocks for lattice-based cryptographic schemes, particularly [ML-DSA (Dilithium)](https://csrc.nist.gov/pubs/fips/204/final). Lattice-based cryptography is one of the leading candidates for post-quantum security: its hardness assumptions (Module-LWE, Module-SIS) are believed to resist attacks by quantum computers.

## Implemented modules

- [`sampling/`](./sampling/mod.rs) — Deterministic SHAKE-based sampling functions following FIPS 204:
  - `expand_a` — generates the public matrix **A** from a seed (Algorithm 32, SHAKE-128)
  - `expand_s` — generates short secret vectors **s** with bounded coefficients (Algorithm 33, SHAKE-256)
  - `sample_challenge` — samples the sparse challenge polynomial **c** with exactly `tau` coefficients in {-1, 0, +1} (Algorithm 29, SampleInBall)
  - `sample_mask` — samples the mask vector **y** with coefficients in `[-gamma_1 + 1, gamma_1 - 1]` (Algorithm 34, SHAKE-256)

## Mathematical background

### Polynomial rings

Dilithium operates in the polynomial ring `Rq = Zq[X]/(X^N + 1)`, where `q = 8380417` and `N = 256`. Elements of this ring are polynomials of degree less than 256 with coefficients modulo `q`. The quotient by `X^N + 1` means that `X^N ≡ -1`, so any polynomial of degree ≥ N wraps around with a sign flip. This negacyclic structure is what makes NTT-based multiplication efficient.

For example, if we multiply two polynomials and get a term `c * X^(N+k)`, the reduction replaces it with `-c * X^k`. This is because `X^N ≡ -1` in the ring, so `X^(N+k) = X^N * X^k ≡ -X^k`.

The ring `Rq` is implemented generically as [`PolynomialRingElement<F, N>`](../../../math/src/polynomial/quotient_ring.rs), which wraps lambdaworks' existing `Polynomial<FieldElement<F>>` and automatically reduces modulo `X^N + 1` on every operation.

### The Dilithium prime

The prime `q = 8380417` is specifically chosen so that `q ≡ 1 (mod 2N)`, which guarantees a primitive `2N`-th root of unity exists in `Zq`, enabling the Number Theoretic Transform (NTT). Concretely:

```
q - 1 = 8380416 = 2^13 × 1023
```

This gives a two-adicity of 13 (`TWO_ADICITY = 13`), meaning the NTT can operate on vectors of size up to `2^13 = 8192`. Since Dilithium only needs `N = 256 = 2^8`, this is more than sufficient.

The field is defined as [`DilithiumField`](../../../math/src/field/fields/fft_friendly/dilithium_prime.rs), a `U64PrimeField<8380417>` implementing `IsFFTField`. The two-adic primitive root of unity is `1938117`, verified to satisfy `1938117^(2^13) ≡ 1 (mod q)`.

### Multiplication: schoolbook vs NTT

`PolynomialRingElement` supports two multiplication algorithms:

- **Schoolbook** (`mul_schoolbook`): Standard polynomial multiplication followed by reduction modulo `X^N + 1`. Complexity is O(N²). Always works for any field.
- **NTT** (`mul_ntt`): Transforms both polynomials to evaluation form via FFT, performs pointwise multiplication, and transforms back. Complexity is O(N log N). Requires the field to implement `IsFFTField` with sufficient two-adicity. If the FFT fails (insufficient two-adicity for the given N), it falls back to schoolbook automatically.

For `DilithiumField` with `N = 256`, the NTT path always succeeds and is significantly faster.

### Short vectors and the infinity norm

The security of lattice schemes relies on the hardness of finding short vectors. "Short" is measured by the infinity norm: the maximum absolute value of any coefficient when mapped to its centered representation.

A coefficient `c` in `[0, q)` is centered by mapping values greater than `(q-1)/2` to their negative equivalent: `c - q`. For `q = 8380417`, this maps `[0, 4190208]` to itself and `[4190209, 8380416]` to `[-4190208, -1]`.

The `PolynomialRingElement` type provides:

- `centered_coefficient(i)` — maps a coefficient from `[0, q)` to `[-(q-1)/2, (q-1)/2]`
- `infinity_norm()` — returns `max |c_i|` in centered form
- `is_small(bound)` — checks whether all `|c_i| ≤ bound`

These are used throughout Dilithium to verify that vectors remain "short enough" for security guarantees. For instance, the secret key vectors **s1**, **s2** must satisfy `is_small(eta)`, and the challenge polynomial **c** has infinity norm exactly 1.

### Sampling

All sampling is deterministic: the same seed always produces the same output. The functions use SHAKE-128/256 as extensible output functions (XOF) and follow FIPS 204 exactly:

- **Rejection sampling** (`expand_a`): draws 3 bytes at a time, forms a 23-bit candidate, rejects if ≥ `q`. Acceptance rate is `q / 2^23 ≈ 99.8%`. Produces the public matrix **A** as a `k × l` grid of ring elements.
- **Bounded coefficient sampling** (`expand_s`): extracts pairs of nibbles from each byte, computes their difference to get values in `[-eta, eta]`. Used for the short secret vectors **s1** and **s2**.
- **SampleInBall** (`sample_challenge`): Fisher-Yates-like shuffle placing exactly `tau` non-zero coefficients (±1). The resulting polynomial has exactly `tau` coefficients that are +1 or -1, and the rest are 0.
- **Mask sampling** (`sample_mask`): bit-packing extraction with 18-bit or 20-bit encoding depending on `gamma_1`. Produces the masking vector **y** used during signing.

Each function takes a 32-byte seed and additional parameters (matrix indices, bounds), ensuring the full Dilithium key generation and signing process is reproducible.

## Code examples

### Ring arithmetic

```rust
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::dilithium_prime::DilithiumField;
use lambdaworks_math::polynomial::quotient_ring::PolynomialRingElement;

type FE = FieldElement<DilithiumField>;
type R256 = PolynomialRingElement<DilithiumField, 256>;

// Create ring elements
let a = R256::new(&[FE::from(1u64), FE::from(2u64), FE::from(3u64)]);
let b = R256::new(&[FE::from(4u64), FE::from(5u64)]);

// Arithmetic
let sum = &a + &b;
let diff = &a - &b;
let neg = -&a;
let scaled = a.scalar_mul(&FE::from(7u64));

// Multiplication (two methods, same result)
let product_school = a.mul_schoolbook(&b);
let product_ntt = a.mul_ntt(&b); // uses FFT, faster for large polynomials
assert_eq!(product_school, product_ntt);

// X^256 ≡ -1 in the ring
let mut coeffs = vec![FE::from(0u64); 257];
coeffs[256] = FE::from(1u64);
let wrapped = R256::new(&coeffs);
assert_eq!(wrapped.coefficient(0), FE::from(8380417 - 1)); // -1 mod q
```

### Sampling

```rust
use lambdaworks_crypto::lattice::sampling::{expand_a, expand_s, sample_challenge};

let seed = [42u8; 32];

// Generate the public matrix A (k=4, l=4 for Dilithium-II)
let a_matrix = expand_a::<256>(&seed, 4, 4);

// Generate short secret vectors with coefficients in [-2, 2]
let s_vectors = expand_s::<256>(&seed, 2, 4);
assert!(s_vectors[0].is_small(2));

// Generate a challenge polynomial with exactly 39 non-zero +-1 coefficients
let c = sample_challenge::<256>(&seed, 39);
assert_eq!(c.infinity_norm(), 1); // all non-zero coefficients are +-1
```

### Centered representation and norms

```rust
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::dilithium_prime::DilithiumField;
use lambdaworks_math::polynomial::quotient_ring::PolynomialRingElement;

type FE = FieldElement<DilithiumField>;
type R256 = PolynomialRingElement<DilithiumField, 256>;

// Coefficients near q are negative in centered form
let p = R256::new(&[FE::from(8380416u64)]); // q - 1
assert_eq!(p.centered_coefficient(0), -1);

// Infinity norm measures the largest absolute coefficient
let p = R256::new(&[FE::from(100u64), FE::from(8380417u64 - 50)]); // [100, -50]
assert_eq!(p.infinity_norm(), 100);
assert!(p.is_small(100));
assert!(!p.is_small(99));
```

## Architecture

The implementation builds on lambdaworks' existing infrastructure rather than reimplementing field/polynomial arithmetic:

- `DilithiumField` is defined as `U64PrimeField<8380417>` with an `IsFFTField` implementation, reusing lambdaworks' generic prime field with compile-time modulus.
- `PolynomialRingElement<F, N>` wraps `Polynomial<FieldElement<F>>`, reusing all existing polynomial arithmetic and FFT-based multiplication.
- The sampling functions use SHAKE-128/256 from the `sha3` crate, which is already a dependency of `lambdaworks-crypto`.

This generic approach means the same `PolynomialRingElement` type works for any field and ring dimension, not just Dilithium's specific parameters. For example, Kyber uses `q = 3329` and `N = 256`, and could reuse the same ring type with a different field.

## References

- [FIPS 204: Module-Lattice-Based Digital Signature Standard (ML-DSA)](https://csrc.nist.gov/pubs/fips/204/final) — the NIST standard (formerly Dilithium)
- [pq-crystals/dilithium](https://github.com/pq-crystals/dilithium) — reference C implementation by the Dilithium team
- [Lattices in Cryptography (Peikert)](https://web.eecs.umich.edu/~cpeikert/lic15/lec01.pdf) — lecture notes on lattice-based crypto foundations
