# Circuit API
In this section, we'll discuss how to build your own constraint system to prove the execution of a particular program.

## Simple Example

Let's take the following simple program as an example. We have two public inputs: `x` and `y`. We want to prove to a verifier that we know a private input `e` such that `x * e = y`. You can achieve this by building the following constraint system:

```rust
use lambdaworks_plonk::constraint_system::ConstraintSystem;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrField;

fn main() {
    let system = &mut ConstraintSystem::<FrField>::new();
    let x = system.new_public_input();
    let y = system.new_public_input();
    let e = system.new_variable();

    let z = system.mul(&x, &e);
    
    // This constraint system asserts that x * e == y
    system.assert_eq(&y, &z);
}
```

This code creates a constraint system over the field of the BLS12381 curve. Then, it creates three variables: two public inputs `x` and `y`, and a private variable `e`. Note that every variable is private except for the public inputs. Finally, it adds the constraints that represent a multiplication and an assertion.

Before generating proofs for this system, we need to run a setup and obtain a verifying key:

```rust
let common = CommonPreprocessedInput::from_constraint_system(&system, &ORDER_R_MINUS_1_ROOT_UNITY);
let srs = test_srs(common.n);
let kzg = KZG::new(srs); // The commitment scheme for plonk.
let vk = setup(&common, &kzg);
```

Now we can generate proofs for our system. We just need to specify the public inputs and obtain a witness that is a solution for our constraint system:

```rust
let inputs = HashMap::from([(x, FieldElement::from(4)), (e, FieldElement::from(3))]);
let assignments = system.solve(inputs).unwrap();
let witness = Witness::new(assignments, &system);
```

Once you have all these ingredients, you can call the prover:

```rust
let public_inputs = system.public_input_values(&assignments);
let prover = Prover::new(kzg.clone(), TestRandomFieldGenerator {});
let proof = prover.prove(&witness, &public_inputs, &common, &vk);
```

and verify:

```rust
let verifier = Verifier::new(kzg);
assert!(verifier.verify(&proof, &public_inputs, &common, &vk));
```

## Building Complex Systems

Some operations are common, and it makes sense to wrap the set of constraints that do these operations in a function and use it several times. Lambdaworks comes with a collection of functions to help you build your own constraint systems, such as conditionals, inverses, and hash functions.

However, if you have an operation that does not come with Lambdaworks, you can easily extend Lambdaworks functionality. Suppose that the exponentiation operation is something common in your program. You can write the [square and multiply](https://en.wikipedia.org/wiki/Exponentiation_by_squaring) algorithm and put it inside a function:

```rust
pub fn pow(
    system: &mut ConstraintSystem<FrField>,
    base: Variable,
    exponent: Variable,
) -> Variable {
    let exponent_bits = system.new_u32(&exponent);
    let mut result = system.new_constant(FieldElement::one());

    for i in 0..32 {
        if i != 0 {
            result = system.mul(&result, &result);
        }
        let result_times_base = system.mul(&result, &base);
        result = system.if_else(&exponent_bits[i], &result_times_base, &result);
    }
    result
}
```

This function can then be used to modify our simple program from the previous section. The following circuit checks that the prover knows `e` such that `pow(x, e) = y`:

```rust
use lambdaworks_plonk::constraint_system::ConstraintSystem;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrField;

fn main() {
    let system = &mut ConstraintSystem::<FrField>::new();
    let x = system.new_public_input();
    let y = system.new_public_input();
    let e = system.new_variable();

    let z = pow(system, &x, &e);
    system.assert_eq(&y, &z);
}
```

You can keep composing these functions in order to create more complex systems.

