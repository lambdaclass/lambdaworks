# GKR Protocol

An implementation of the Goldwasser-Kalai-Rothblum (GKR) Non-Interactive Protocol for proving correct evaluation of arithmetic circuits.

To help with the understanding of this implementation, we recommend reading our [blog post](https://blog.lambdaclass.com/gkr-protocol-a-step-by-step-example/).

**Warning:** This GKR implementation is for educational purposes and should not be used in production. It uses the Fiat-Shamir transform, which is vulnerable to practical attacks in this context (see ["How to Prove False Statments"](https://eprint.iacr.org/2025/118.pdf)). 

## Overview

The GKR Protocol allows a Prover to convince a Verifier that he correctly evaluated an arithmetic circuit on a given input without the Verifier having to perform the entire computation.   

It is a fundamental building block for many interactive proof systems and argument systems, providing a way to verify computations in time roughly proportional to the circuit's depth rather than its size.

The protocol works by reducing claims about each layer of the circuit to claims about the next layer, using the Sumcheck Protocol as a subroutine. This process continues until reaching the input layer, where the verifier can directly check the final claim. The key insight is that the wiring of the circuit can be expressed as multilinear polynomials, allowing the use of sumcheck for efficient verification.

### Key Features

- **Layered Circuit Support**: Works with circuits organized in layers where each gate takes inputs from the previous layer.
- **Power-of-Two Constraint**: Each layer must have a power-of-two number of gates for protocol compatibility.
- **Addition and Multiplication Gates**: Supports both addition and multiplication operations.
- **Complete Verification**: Includes input verification to ensure end-to-end correctness.

## Circuit Structure

Circuits are organized in layers, with each layer containing gates that operate on outputs from the previous layer:

```
Output:    o          o      <- circuit.layers[0]
         /   \      /   \
        o     o    o     o   <- circuit.layers[1]
                ...        
           o   o   o   o     <- circuit.layers[layer.len() - 1]
          / \ / \ / \ / \
Input:    o o o o o o o o
```

Each layer must have a power-of-two number of gates.

## API

### Main Functions

- `gkr_prove(circuit, input)` - Generate a GKR proof for a circuit evaluation.
- `gkr_verify(proof, circuit, input)` - Verify a GKR proof.

### Circuit Construction

- `Circuit::new(layers, num_inputs)` - Create a new circuit with specified layers, gates and number of inputs.
- `CircuitLayer::new(gates)` - Create a layer with the given gates.
- `Gate::new(gate_type, inputs_idx)` - Create a gate with type (Add/Mul) and certain input indeces.

## Example

Here's a simple example of how to use the GKR Protocol:

```rust
use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_gkr::{gkr_prove, gkr_verify, circuit_from_lambda};

// Define the field (We use modulus 23 as in the example of our blog post).
const MODULUS23: u64 = 23;
type F23 = U64PrimeField<MODULUS23>;
type F23E = FieldElement<F23>;

// Create the circuit of our blog post.
// This creates a 2-layer circuit plus the input layer.
let circuit = lambda_post_circuit.unwrap();

// Define input values (from the post example).
let input = [F23E::from(3), F23E::from(1)];

// Generate proof.
let proof_result = gkr_prove(&circuit, &input);
assert!(proof_result.is_ok());
let proof = proof_result.unwrap();

// Verify proof.
let verification_result = gkr_verify(&proof, &circuit, &input);
assert!(verification_result.is_ok() && verification_result.unwrap());
println!("GKR verification successful!");

// You can also check the actual output.
let evaluation = circuit.evaluate(&input);
println!("Circuit output: {:?}", evaluation.layers[0]);
```

### Creating Custom Circuits

You can create custom circuits by defining your own layers, gates and number of inputs.:

```rust
use lambdaworks_gkr::circuit::{Circuit, CircuitLayer, Gate, GateType};

// Create a simple 2-layer circuit
let custom_circuit = Circuit::new(
    vec![
        // Output layer: 1 gate
        CircuitLayer::new(vec![
            Gate::new(GateType::Add, [0, 1]),  // Add the two results from layer 1
        ]),
        // Layer 1: 2 gates
        CircuitLayer::new(vec![
            Gate::new(GateType::Mul, [0, 1]),  // Multiply first two inputs
            Gate::new(GateType::Mul, [2, 3]),  // Multiply last two inputs
        ]),
    ],
    4, // 4 inputs (power of 2)
).unwrap();

// Test with inputs [2, 3, 4, 5]
let input = [F23E::from(2), F23E::from(3), F23E::from(4), F23E::from(5)];

let proof = gkr_prove(&custom_circuit, &input).unwrap();
let is_valid = gkr_verify(&proof, &custom_circuit, &input).unwrap();

assert!(is_valid);
```

## Protocol Details

The GKR protocol works through the following steps:

1. **Circuit Evaluation**: The prover evaluates the circuit on the given input to obtain the values at each layer.
2. **Layer-by-Layer Reduction**: For each layer $i$, the prover uses the Sumcheck Protocol to reduce a claim about the current layer to a claim about the next layer $i+1$.

    - **Wiring Polynomial Construction**: The circuit's wiring is encoded as multilinear polynomials $\widetilde{\text{add}_i}$ and $\widetilde{\text{mul}_i}$ that describe which gates are addition/multiplication gates.

    - **Sumcheck Application**: The sumcheck is applied to the polynomial:
        $$\tilde f_{r_i}(b,c) = \widetilde{\text{add}_i(}r_i, b, c) \cdot (\tilde W_{i+1}(b) + \tilde W_{i+1}(c)) + \widetilde{\text{mul}_i}(r_i, b, c) \cdot (\tilde W_{i+1}(b) \cdot \tilde W_{i+1}(c))$$

        where $\tilde W_{i+1}$ is the multilinear extension of layer $i+1$ values.

    - **Line Function**: A line function transforms the two  claims of $\tilde W_{i+1}(b)$ and $\tilde W_{i+1}(c)$ into a single claim.
3. **Input Verification**: The verifier checks the final claim evaluating the input multilinear polynomial extension at the final evaluation point.

Each *layer proof* consists of:
- The claimed sum (that the sumcheck proves).
- All the univariate polynomials used in the sumcheck. They are built fixing the first variable and summing over the rest of them. 
- The polynomial $q = \tilde W_{i+1} \circ \ell$ where $\ell$ is the line function.

The protocol achieves $O(d \log S)$ verifier time and $O(S)$ prover time, where $d$ is the circuit depth and $S$ is the circuit size (i.e. the number of gates).


## Fiat-Shamir transform

This implementation uses  **Fiat-Shamir** to transform the interactive GKR protocol into a non-interactive proof system. Instead of requiring back-and-forth communication between prover and verifier, the prover generates all challenges deterministically using a cryptographic transcript.

### How Fiat-Shamir is Applied

The transformation works by replacing the verifier's random challenges with outputs from a cryptographic hash function (transcript):

1. **Transcript Initialization**: A transcript is created and seeded with:
   - Circuit structure (via `circuit_to_bytes(circuit)`)
   - Input values 
   - Output values

2. **Challenge Generation**: At each step where the interactive protocol would require a random challenge, the implementation:
   - Adds the current proof data to the transcript.
   - Samples a "random" field element from the transcript using `transcript.sample_field_element()`.

3. **Key Challenge Points**:
   - **Initial random values** `r_0` for the output layer.
   - **Sumcheck challenges** ($s_j$) for each round of each layer's sumcheck protocol.
   - **Line function parameter** `r_last` for connecting layers.


## References

- [Goldwasser, Kalai, and Rothblum. "Delegating computation: interactive proofs for muggles"](https://dl.acm.org/doi/10.1145/1374376.1374396)
- [Proofs, Arguments, and Zero-Knowledge. Chapter 4](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)
- [Lambdaclass Blog Post: GKR protocol: a step-by-step example](https://blog.lambdaclass.com/gkr-protocol-a-step-by-step-example/) 
