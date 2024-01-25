# lambdaworks examples

This folder contains examples designed to learn how to use the different tools in lambdaworks, such as finite field arithmetics, elliptic curves, provers, and adapters.

Below is a list of all lambdaworks examples in the folder:
- Merkle tree CLI: generate inclusion proofs for an element inside a Merkle tree and verify them using a CLI
- Proving Miden using lambdaworks STARK Platinum prover: Executes a Miden vm Fibonacci program, gets the execution trace and generates a proof (and verifies it) using STARK Platinum.

You can also check [lambdaworks exercises](https://github.com/lambdaclass/lambdaworks/tree/main/exercises) to learn more.

## Basic use of Finite Fields

This library works with [finite fields](https://en.wikipedia.org/wiki/Finite_field). A `Field` is an abstract definition. It knows the modulus and defines how the operations are performed.

We usually create a new `Field` by instantiating an optimized backend. For example, this is the definition of the Pallas field:

```rust
// 4 is the number of 64-bit limbs needed to represent the field
type PallasMontgomeryBackendPrimeField<T> = MontgomeryBackendPrimeField<T, 4>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MontgomeryConfigPallas255PrimeField;
impl IsModulus<U256> for MontgomeryConfigPallas255PrimeField {
    const MODULUS: U256 = U256::from_hex_unchecked(
        "40000000000000000000000000000000224698fc094cf91b992d30ed00000001",
    );
}

pub type Pallas255PrimeField =
    PallasMontgomeryBackendPrimeField<MontgomeryConfigPallas255PrimeField>;
```

Internally, it resolves all the constants needed and creates all the required operations for the field.

Suppose we want to create a `FieldElement`. This is as easy as instantiating the `FieldElement` over a `Field` and calling a `from_hex` function.

For example:

```rust
 let an_element = FieldElement::<Stark252PrimeField>::from_hex_unchecked("030e480bed5fe53fa909cc0f8c4d99b8f9f2c016be4c41e13a4848797979c662")
```

Notice we can alias the `FieldElement` to something like

```rust
type FE = FieldElement::<Stark252PrimeField>;
```

Once we have a field, we can make all the operations. We usually suggest working with references, but copies work too.

```rust
let field_a = FE::from_hex("3").unwrap();
let field_b = FE::from_hex("7").unwrap();

// We can use pointers to avoid copying the values internally
let operation_result = &field_a * &field_b

// But all the combinations of pointers and values works
let operation_result = field_a * field_b
```

Sometimes, optimized operations are preferred. For example,

```rust
// We can make a square multiplying two numbers
let squared = field_a * field_a;
// Using exponentiation
let squared = 
field_a.pow(FE::from_hex("2").unwrap())
// Or using an optimized function
let squared = field_a.square()
```

ome useful instantiation methods are also provided for common constants and whenever const functions can be called. This is when creating functions that do not rely on the `IsField` trait since Rust does not support const functions in traits yet,

```rust
// Defined for all field elements
// Efficient, but nonconst for the compiler
let zero = FE::zero() 
let one = FE::one()

// Const alternatives of the functions are provided, 
// But the backend needs to be known at compile time. 
// This requires adding a where clause to the function

let zero = F::ZERO
let one = F::ONE
let const_intstantiated = FE::from_hex_unchecked("A1B2C3");
```

You will notice traits are followed by an `Is`, so instead of accepting something of the form `IsField`, you can use `IsPrimeField` and access more functions. The most relevant is `.representative()`. This function returns a canonical representation of the element as a number, not a field.
