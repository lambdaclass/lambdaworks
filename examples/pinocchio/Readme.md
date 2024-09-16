# Lambdaworks Pinocchio 

This is an implementation of [Pinocchio protocol](https://eprint.iacr.org/2013/279.pdf) using Lambdaworks. This source code is the companion of this [blog post](https://blog.lambdaclass.com/pinocchio-verifiable-computation-revisited) aimed at those who want to learn about SNARKs.

# Understanding the code
We encourage to start by reading the [blog post](https://blog.lambdaclass.com/pinocchio-verifiable-computation-revisited/) to understand the code.

Then, in the `tests/` module you will find integration tests that will guide you through the happy path: the setup, the prover and the verifier. Each of these components can be located in their own files:

- `pinocchio/setup.rs`: generates the `VerificationKey` and the `EvaluationKey` with the relevant data to construct and verify proofs. This data comes from the structure of the program P encoded as `Polynomials` in a `QAP` (Quadratic arithmetic program). To hide this data, random `FieldElement`s are sampled as `ToxicWaste` and then mapped to `G1Point`'s and  `G2Point`'s via repeated addition of the `generator()`'s of the curves. 

- `pinocchio/prover.rs`: takes the circuit encoded as a `QAP` and the trace of the program as `FieldElement`s. Then, it applies the `msm(...)` operation in order to generate the proof elements, in this case, `G1Point` and `G2Point` hidings.

- `pinocchio/verifier.rs`: verifies the proof by checking the conditions mentioned in the paper. This involves computing `Pairing::compute(...)` operations between a `G1Point`and a `G2Point`.

# Example

**Step 1:** Construct a QAP

```rust
let test_qap = new_test_r1cs().into();
```

**Step 2**: Setup providing Evaluation and Verification keys.

```rust 
let (evaluation_key, verification_key) = setup(&test_qap, toxic_waste);
```
**Step 3:** The prover constructs the proof using the evaluation key, the QAP and the inputs.
```rust
let proof = generate_proof(&evaluation_key, &test_qap, &c_vector);
```
**Step 4:** The verifier checks the validity of the proof using the verification key and the public inputs.
```rust
let accepted = verify(&verification_key, &proof, &c_io_vector);
```
