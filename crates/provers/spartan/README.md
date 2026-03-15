# Spartan

An implementation of the [Spartan](https://eprint.iacr.org/2019/550) zkSNARK prover for R1CS satisfiability.

**Warning:** This implementation is for educational purposes and should not be used in production.

## Overview

Spartan is a zkSNARK that proves knowledge of a witness satisfying an R1CS instance without a trusted setup. The prover's work is $O(n)$ field operations where $n$ is the number of R1CS constraints, making it one of the most efficient transparent SNARKs for general circuits.

The protocol combines two phases of the [Sumcheck Protocol](../sumcheck/) with a multilinear Polynomial Commitment Scheme (PCS) to produce a succinct proof. The verifier only needs to evaluate three sparse matrix polynomials and verify a single PCS opening, achieving $O(\sqrt{n})$ verification time.

### Key Features

- **Transparent**: No trusted setup required; all randomness is derived via Fiat-Shamir.
- **R1CS-native**: Directly proves $(Az) \cdot (Bz) = Cz$ without conversion to other arithmetizations.
- **Modular PCS**: The prover is generic over any `IsMultilinearPCS` implementation.
- **Zeromorph PCS**: Ships with a [Zeromorph](https://eprint.iacr.org/2023/917) backend backed by BLS12-381 KZG for production-quality polynomial commitments.

## API

### Main Functions

- `spartan_prove(r1cs, public_inputs, witness, pcs)` — Generate a Spartan proof.
- `spartan_verify(r1cs, public_inputs, proof, pcs)` — Verify a Spartan proof.

### PCS Backends

- `ZeromorphPCS<N, F, P>` — Multilinear PCS via Zeromorph reduction to univariate KZG. Backed by any `IsPairing` curve; ships ready to use with BLS12-381.
- `TrivialPCS` — Sends the full witness in the clear. Useful for testing the sumcheck layer in isolation.

## Example

```rust
use lambdaworks_math::field::{
    element::FieldElement,
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};
use lambdaworks_spartan::{
    r1cs::R1CS,
    prover::spartan_prove,
    verifier::spartan_verify,
    pcs::trivial::TrivialPCS,
};

type F = Stark252PrimeField;
type FE = FieldElement<F>;

// Build R1CS for: a * b = c  (witness = [1, c, a, b])
let zero = FE::zero();
let one = FE::one();
let r1cs = R1CS::new(
    vec![vec![zero, zero, one, zero]],   // A: picks a
    vec![vec![zero, zero, zero, one]],   // B: picks b
    vec![vec![zero, one, zero, zero]],   // C: picks c
    1, // 1 public input (c)
).unwrap();

let witness = vec![one, FE::from(6), FE::from(2), FE::from(3)];
let public_inputs = vec![FE::from(6)];

let pcs = TrivialPCS::default();
let proof = spartan_prove(&r1cs, &public_inputs, &witness, pcs.clone()).unwrap();
let ok = spartan_verify(&r1cs, &public_inputs, &proof, pcs).unwrap();
assert!(ok);
```

### With Zeromorph PCS (BLS12-381)

```rust
use lambdaworks_crypto::commitments::kzg::KateZaveruchaGoldberg;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::{
    default_types::FrField,
    pairing::BLS12381AtePairing,
};
use lambdaworks_spartan::pcs::zeromorph::ZeromorphPCS;

type Kzg = KateZaveruchaGoldberg<FrField, BLS12381AtePairing>;
type Pcs = ZeromorphPCS<4, FrField, BLS12381AtePairing>;

// Build a structured reference string (SRS) with enough powers.
// srs_size >= 2^(log2(num_variables) + 1)
let srs = Kzg::create_srs(toxic_waste, srs_size);
let pcs = Pcs::new(Kzg::new(srs));

let proof = spartan_prove(&r1cs, &public_inputs, &witness, pcs.clone()).unwrap();
let ok = spartan_verify(&r1cs, &public_inputs, &proof, pcs).unwrap();
assert!(ok);
```

## Protocol Details

### Phase 1 — Outer Sumcheck

The prover encodes R1CS satisfiability as a sum over the boolean hypercube. For a random point $\tau \in \mathbb{F}^{\log m}$ (derived from the transcript), the prover runs a sumcheck on:

$$\sum_{x \in \{0,1\}^{\log m}} \widetilde{eq}(\tau, x) \cdot \left(\tilde{A}(x, \cdot) \cdot z\right) \left(\tilde{B}(x, \cdot) \cdot z\right) - \left(\tilde{C}(x, \cdot) \cdot z\right) = 0$$

This reduces to three claimed evaluations $v_A$, $v_B$, $v_C$ of the matrix-times-witness products at a random point $r_x$.

### Phase 2 — Inner Sumcheck

The prover batches the three matrix polynomials $\tilde{A}$, $\tilde{B}$, $\tilde{C}$ with random scalars $\rho_A$, $\rho_B$, $\rho_C$ and runs a second sumcheck to reduce the combined claim to a single evaluation of the witness multilinear extension $\tilde{z}(r_y)$.

### PCS Opening

The prover opens $\tilde{z}$ at the inner sumcheck point $r_y$ using the provided multilinear PCS. The verifier recomputes the matrix evaluations $\tilde{A}(r_x, r_y)$, $\tilde{B}(r_x, r_y)$, $\tilde{C}(r_x, r_y)$ directly and checks consistency.

### Zeromorph PCS

The Zeromorph backend (Kohrita & Towa, 2023) reduces a multilinear evaluation claim to a univariate KZG opening. Given $f(u_0, \ldots, u_{n-1}) = v$, it produces quotient polynomials $\hat{q}_0, \ldots, \hat{q}_{n-1}$ via a halving algorithm and batches them into a single KZG proof, achieving $O(n)$ prover work and $O(n)$ verifier work in group operations.

## References

- [Spartan: Efficient and general-purpose zkSNARKs without trusted setup — Setty (2020)](https://eprint.iacr.org/2019/550)
- [Zeromorph: Zero-Knowledge Multilinear-Evaluation Proofs from Homomorphic Univariate Commitments — Kohrita & Towa (2023)](https://eprint.iacr.org/2023/917)
- [Proofs, Arguments, and Zero-Knowledge — Thaler (2022)](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)
