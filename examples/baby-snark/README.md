# Lambdaworks Baby SNARK

An implementation of [Baby SNARK](https://github.com/initc3/babySNARK/blob/bebb2948f8094a8d3949afe6d10b89a120a005be/babysnark.pdf) protocol.

# Example

Below is a simple example demonstrating the usage of BabySnark:

**Step 1:** Construct Span Program (ssp):

```rust
    // Define Constraint Matrix 
    let u = vec![
        i64_vec_to_field(&[-1, 2, 0, 0]),
        i64_vec_to_field(&[-1, 0, 2, 0]),
        i64_vec_to_field(&[-1, 0, 0, 2]),
        i64_vec_to_field(&[-1, 2, 2, -4]),
    ];
    let witness = i64_vec_to_field(&[1, 1, 1]);
    let public = i64_vec_to_field(&[1]);
    let mut input = public.clone();
    input.extend(witness.clone());
    // Construct Span Program (ssp):
    let ssp = SquareSpanProgram::from_scs(SquareConstraintSystem::from_matrix(u, public.len()));
```

**Step 2:** Setup Proving and Verification Keys:
```rust
    let (pk, vk) = setup(&ssp);
```
**Step 3:** The prover generates a proof using the input, the SSP, and the proving key.
```rust
    let proof = Prover::prove(&input, &ssp, &pk);
```
**Step 4:** The verifier checks the validity of the proof using the verification key and the public input.
```rust
    let verified = verify(&vk, &proof, &public);
    assert!(verified);
```