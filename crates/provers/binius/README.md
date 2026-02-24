# Lambdaworks Binius Prover

An implementation of the [Binius](https://eprint.iacr.org/2023/1786) proof system, a SNARK designed for native binary field arithmetic. This is part of the [Lambdaworks](https://github.com/lambdaclass/lambdaworks) zero-knowledge framework.

> **Warning:** This prover is still in development and may contain bugs. It is not intended to be used in production yet.

## Overview

Binius is a proof system built entirely over binary fields — towers of extensions of GF(2). Unlike most SNARKs that operate over large prime fields (BN254, BLS12-381, Goldilocks), Binius works with the same field that computers naturally compute in: binary. This eliminates the overhead of embedding binary computations into prime field arithmetic.

The key insight is that binary towers GF(2) ⊂ GF(2²) ⊂ GF(2⁴) ⊂ ... ⊂ GF(2¹²⁸) provide a rich algebraic structure where:
- **Addition is XOR** — the cheapest operation a CPU can do
- **Small values stay small** — a single bit lives in GF(2), a byte in GF(2⁸), no need to embed into a 256-bit prime field
- **Bitwise operations are native** — AND, XOR, shifts are field operations, not expensive gadgets

### Why Binius matters

In a typical STARK or SNARK, proving that two 32-bit integers were AND-ed correctly requires encoding each bit as a full field element (e.g., a 256-bit number), applying constraints bit by bit, and paying the cost of large-field arithmetic at every step. Binius avoids this entirely: a 32-bit AND is a single field multiplication in GF(2³²).

This makes Binius particularly well-suited for:
- Hash function verification (SHA-256, Keccak)
- Bitwise operations (AES, bit manipulation)
- Any computation that is naturally binary

## Architecture

The prover is built in several layers:

```
┌─────────────────────────────────────────────┐
│              Constraint System              │  Circuit definition (gates, wires)
│         (AND, XOR, MUL, SELECT, ...)        │
├─────────────────────────────────────────────┤
│              Binius Prover                  │  Composes sumcheck + FRI
│   prove_polynomial() / verify_proof()       │
├──────────────────────┬──────────────────────┤
│     Sum-check        │     Binary FRI       │  Core protocols
│  (product sumcheck)  │ (additive folding)   │
├──────────────────────┴──────────────────────┤
│            Additive NTT                     │  RS encoding over binary subspaces
│     (evaluation on GF(2)-linear subspaces)  │
├─────────────────────────────────────────────┤
│         Binary Tower Field                  │  GF(2^128) via tower construction
│    (BinaryTowerField128 implements IsField) │
└─────────────────────────────────────────────┘
```

| Module | Role |
|--------|------|
| `fields::tower` | Tower field arithmetic (GF(2) up to GF(2¹²⁸)) |
| `ntt` | Additive NTT: polynomial evaluation over GF(2)-linear subspaces |
| `fri` | Binary FRI: additive folding commitment scheme |
| `sumcheck` | Sum-check protocol (delegates to `lambdaworks-sumcheck`) |
| `prover` | Unified prover composing sumcheck + FRI with Fiat-Shamir |
| `verifier` | Verifier with real checks (transcript replay, sumcheck, FRI) |
| `constraints` | Gate-level circuit representation and witness generation |
| `merkle` | Merkle tree for FRI commitments |

## Protocol

Given a witness polynomial $f: \{0,1\}^n \to \text{GF}(2^{128})$, the Binius protocol proves knowledge of $f$ through the following steps:

### 1. Commit

The prover Reed-Solomon encodes $f$ by evaluating it on a larger GF(2)-linear subspace (using the additive NTT), then Merkle-commits the resulting codeword.

### 2. Derive evaluation point

Using the Fiat-Shamir heuristic, a random evaluation point $\mathbf{r} = (r_1, \ldots, r_n) \in \text{GF}(2^{128})^n$ is derived from the commitment.

### 3. Evaluation proof via sum-check

The prover computes $v = f(\mathbf{r})$ and proves this evaluation using the identity:

$$f(\mathbf{r}) = \sum_{\mathbf{x} \in \{0,1\}^n} \text{eq}(\mathbf{x}, \mathbf{r}) \cdot f(\mathbf{x})$$

where $\text{eq}(\mathbf{x}, \mathbf{r}) = \prod_{i=1}^{n} (x_i r_i + (1 - x_i)(1 - r_i))$ is the multilinear equality polynomial.

This sum is proved via a product sumcheck over the two multilinear polynomials $\text{eq}(\cdot, \mathbf{r})$ and $f(\cdot)$.

### 4. Low-degree test via binary FRI

The FRI protocol proves that the committed codeword is close to a low-degree polynomial. Binary FRI differs from standard multiplicative FRI in the folding mechanism:

Given $f(x)$ evaluated on a GF(2)-linear subspace $W$, we decompose:

$$f(x) = f_0(V(x)) + x \cdot f_1(V(x))$$

where $V(x) = x^2 + x$ is the vanishing polynomial of $\{0, 1\} \subset W$.

With a random challenge $\alpha$, the folded polynomial is:

$$f_\alpha(V(x)) = f_0(V(x)) + \alpha \cdot f_1(V(x))$$

which has half the degree of $f$. Crucially, the folded evaluation domain transforms via $V(x) = x^2 + x$ at each round — the domain points are **not** simple integers after the first fold.

### 5. Verification

The verifier:
1. Replays the Fiat-Shamir transcript to derive the same evaluation point
2. Checks sumcheck round polynomial consistency ($g_1(0) + g_1(1) = \text{claimed\_sum}$, etc.)
3. Verifies the FRI proof (folding equations, Merkle proofs, final constant check)
4. Checks that the sumcheck claimed sum equals the claimed evaluation $f(\mathbf{r})$

## Examples

### Direct polynomial proof

The simplest way to use Binius is to prove knowledge of a multilinear polynomial:

```rust
use lambdaworks_binius::fri::FriParams;
use lambdaworks_binius::prover::BiniusProver;
use lambdaworks_binius::verifier::BiniusVerifier;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::binary::tower_field::BinaryTowerField128;
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;

type FE = FieldElement<BinaryTowerField128>;

// Create a 2-variable polynomial: f(x,y) with 4 evaluations
let evals = vec![
    FE::new(5u128),
    FE::new(3u128),
    FE::new(7u128),
    FE::new(11u128),
];
let witness = DenseMultilinearPolynomial::<BinaryTowerField128>::new(evals);

// Set up FRI parameters
let fri_params = FriParams {
    log_message_size: 2,  // 2^2 = 4 evaluations
    log_blowup: 1,        // 2x blowup for RS encoding
    num_queries: 2,        // number of query positions
};

// Prove
let prover = BiniusProver::new(fri_params.clone());
let proof = prover.prove_polynomial(&witness).unwrap();

// Verify
let verifier = BiniusVerifier::new(fri_params);
assert!(verifier.verify_proof(&proof).is_ok());
```

### Circuit-based proof

For more complex computations, use the constraint system to define a circuit, execute it to get a witness, and prove:

```rust
use lambdaworks_binius::constraints::ConstraintSystem;
use lambdaworks_binius::fields::tower::Tower;
use lambdaworks_binius::fri::FriParams;
use lambdaworks_binius::prover::BiniusProver;
use lambdaworks_binius::verifier::BiniusVerifier;

// Define a circuit: c = AND(a, b), d = MUL(a, c)
let mut cs = ConstraintSystem::new();
let a = cs.new_byte();
let b = cs.new_byte();
let c = cs.and(a, b);
let _d = cs.mul(a, c);

// Execute with concrete inputs: a = 0xFF, b = 0x0F
let witness = cs.execute(&[Tower::new(0xFF, 3), Tower::new(0x0F, 3)]);

// Convert witness to a multilinear polynomial
let poly = cs.witness_to_dense_poly(&witness);

// Prove and verify
let fri_params = FriParams {
    log_message_size: poly.num_vars(),
    log_blowup: 1,
    num_queries: 2,
};
let prover = BiniusProver::new(fri_params.clone());
let verifier = BiniusVerifier::new(fri_params);

let proof = prover.prove_polynomial(&poly).unwrap();
assert!(verifier.verify_proof(&proof).is_ok());
```

### Supported gates

The constraint system supports the following operations:

```rust
let mut cs = ConstraintSystem::new();

// Variables at different field levels
let a = cs.new_word();       // 64-bit (GF(2^64))
let b = cs.new_word();
let c = cs.new_byte();       // 8-bit  (GF(2^8))
let cond = cs.new_byte();

// Gates
let and_result = cs.and(a, b);          // Bitwise AND
let xor_result = cs.xor(a, b);          // Bitwise XOR
let mul_result = cs.mul(a, b);          // Field multiplication
let rot_result = cs.rot(a, 3);          // Rotation by 3 bits
let sel_result = cs.select(cond, a, b); // if cond != 0 { a } else { b }
let constant   = cs.constant(Tower::new(42, 3));

// Assertions
cs.assert_eq(a, b);  // Enforce a == b
```

### Tamper detection

The verifier rejects tampered proofs:

```rust
let mut proof = prover.prove_polynomial(&witness).unwrap();

// Tamper with the claimed evaluation
proof.claimed_evaluation = FE::new(999u128);

// Verification fails
assert!(verifier.verify_proof(&proof).is_err());
```

## Binary tower field

The foundation of Binius is the binary tower field GF(2¹²⁸), implemented as `BinaryTowerField128`. It implements the `IsField` trait, which unlocks all existing lambdaworks infrastructure:

- `DenseMultilinearPolynomial<BinaryTowerField128>` for multilinear polynomials
- `DefaultTranscript<BinaryTowerField128>` for Fiat-Shamir
- `Polynomial<FieldElement<BinaryTowerField128>>` for univariate polynomials

The tower is constructed recursively:

| Level | Field | Size | Example use |
|-------|-------|------|-------------|
| 0 | GF(2) | 1 bit | Single bits |
| 1 | GF(2²) = GF(4) | 2 bits | - |
| 3 | GF(2⁸) = GF(256) | 1 byte | AES S-box |
| 5 | GF(2³²) | 4 bytes | Hash words |
| 6 | GF(2⁶⁴) | 8 bytes | Machine words |
| 7 | GF(2¹²⁸) | 16 bytes | Proof field |

At each level $k$, GF(2^{2^k}) is constructed as a degree-2 extension of GF(2^{2^{k-1}}). Arithmetic in characteristic 2 means:
- Addition = XOR (no carries)
- Negation = identity ($-a = a$)
- Subtraction = XOR (same as addition)

## Additive NTT

Unlike the standard multiplicative FFT used in prime-field SNARKs, Binius uses an **additive NTT** that evaluates polynomials over GF(2)-linear subspaces.

The evaluation domain is the subspace $W = \text{span}(e_0, e_1, \ldots, e_{k-1})$ where $e_i = 2^i$ are the canonical GF(2)-basis vectors. This gives the domain $\{0, 1, 2, \ldots, 2^k - 1\}$ viewed as field elements.

Key operations:
- `additive_ntt(coeffs, log_size)` — evaluate polynomial at all subspace points
- `inverse_additive_ntt(evals, log_size)` — interpolate from evaluations
- `rs_encode(message, log_blowup)` — Reed-Solomon encoding on a larger subspace
- `fold_codeword(codeword, challenge, domain)` — FRI folding with domain tracking

## References

### Papers

- [Binius: Efficient Proofs over Binary Fields](https://eprint.iacr.org/2023/1786) — the original Binius paper by Ulvetanna
- [FRI-Binius: Polylogarithmic Proofs for Large Binary Fields](https://eprint.iacr.org/2024/504) — FRI adaptation for binary fields
- [Novel Polynomial Basis and Its Application to Reed-Solomon Erasure Codes](https://arxiv.org/pdf/1708.09746) — LCH14, the additive FFT foundation

### Blog posts

- [Binius — Vitalik Buterin](https://vitalik.eth.limo/general/2024/04/29/binius.html) — accessible explanation of Binius
- [Additive FFT Background — LambdaClass](https://blog.lambdaclass.com/additive-fft-background/) — additive NTT foundations
- [Have You Checked Your Sums? — LambdaClass](https://blog.lambdaclass.com/have-you-checked-your-sums/) — sumcheck protocol

### Implementations

- [binius-zk/binius64](https://github.com/binius-zk/binius64) — reference Binius implementation
- [IrreducibleOSS/binius](https://github.com/IrreducibleOSS/binius) — Irreducible's Binius implementation
