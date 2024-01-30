# Lambdaworks Plonk Prover
A fast implementation of the [Plonk](https://eprint.iacr.org/2019/953) zk-protocol written in Rust. This is part of the [Lambdaworks](https://github.com/lambdaclass/lambdaworks) zero-knowledge framework. It includes a high-level API to seamlessly build your own circuits.

This prover is still in development and may contain bugs. It is not intended to be used in production yet.

## Building a circuit

Starting with an example, the following code creates a circuit with two public inputs `x`, `y` and asserts `x * e = y`:

```rust
let system = &mut ConstraintSystem::<FrField>::new();
let x = system.new_public_input();
let y = system.new_public_input();
let e = system.new_variable();

let z = system.mul(&x, &e);    
system.assert_eq(&y, &z);;
```

By placing this logic under one function, one can create "gadgets" to abstract functionality.

```Rust
/// A square and multiply implementation.
pub fn pow<F: IsPrimeField>(
    system: &mut ConstraintSystem<F>,
    base: Variable,
    exponent: Variable,
) -> Variable {
    let exponent_bits = system.new_u32(&exponent);
    let mut result = system.new_constant(FE::one());

    assert_eq!(exponent_bits.len(), 32);
    for (i, bit) in exponent_bits.iter().enumerate() {
        if i != 0 {
            result = system.mul(&result, &result);
        }
        let result_times_base = system.mul(&result, &base);
        result = system.if_else(bit, &result_times_base, &result);
    }
    result
}
```

The core operations supported by plonk and our prove system are:

```rust
mul(var1,var2)
add(var1,var2)
add_constant(var1,constant)
div(var1,var2)
// c1 * v1 + c2 * v2 + b = w
// hinted value can be w,v1, or v2
let w = linear_combination(&v1, c1, &v2, c2, b, Option(hint))
```

All the variables and constants are finite fields. Abstractions like integers are not implemented yet.

## Generating a proof
### Setup
A setup is needed in order to generate a proof for a new circuit. The following code generates a verifying key that will be used by both the prover and the verifier:

```rust
let common = CommonPreprocessedInput::from_constraint_system(&system, &ORDER_R_MINUS_1_ROOT_UNITY);
let srs = test_srs(common.n);
let kzg = KZG::new(srs); // The commitment scheme for plonk.
let verifying_key = setup(&common, &kzg);
```

### Prover
First, we fix values for `x` and `e` and solve the constraint system:
```rust
let inputs = HashMap::from([(x, FieldElement::from(4)), (e, FieldElement::from(3))]);
let assignments = system.solve(inputs).unwrap();
```

Finally, we call the prover:
```rust
let witness = Witness::new(assignments, &system);
let public_inputs = system.public_input_values(&assignments);
let prover = Prover::new(kzg.clone(), TestRandomFieldGenerator {});
let proof = prover.prove(&witness, &public_inputs, &common, &verifying_key);
```

## Verifying a proof
Just call the verifier:

```rust
let verifier = Verifier::new(kzg);
assert!(verifier.verify(&proof, &public_inputs, &common, &verifying_key));
```

# More info
You can find more info in the [documentation](https://lambdaclass.github.io/lambdaworks_plonk_prover/).
