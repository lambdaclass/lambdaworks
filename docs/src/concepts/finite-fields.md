# Finite Fields

Finite fields are the mathematical foundation of all cryptographic operations in lambdaworks. This document explains the concepts and shows how to work with fields in the library.

## What is a Finite Field?

A finite field (also called a Galois field) is a set of elements with a finite number of members, together with two operations (addition and multiplication) that satisfy the standard field axioms. Unlike real numbers, finite fields have a limited number of elements.

The simplest finite fields are prime fields, denoted $\mathbb{F}_p$, where $p$ is a prime number. The elements of $\mathbb{F}_p$ are the integers $\{0, 1, 2, \ldots, p-1\}$, and all arithmetic is performed modulo $p$.

For example, in $\mathbb{F}_7$:
- $5 + 4 = 9 \equiv 2 \pmod{7}$
- $3 \times 4 = 12 \equiv 5 \pmod{7}$
- $3^{-1} = 5$ because $3 \times 5 = 15 \equiv 1 \pmod{7}$

## Fields in lambdaworks

lambdaworks represents field elements using the `FieldElement<F>` type, where `F` is a type implementing the `IsField` trait. This generic design allows the same code to work with any field.

### Creating Field Elements

```rust
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

// Create a type alias for convenience
type FE = FieldElement<Stark252PrimeField>;

// From integers
let a = FE::from(42u64);
let b = FE::from(7u64);

// From hex strings
let c = FE::from_hex_unchecked("0x1a2b3c");

// Special values
let zero = FE::zero();
let one = FE::one();
```

### Arithmetic Operations

All standard arithmetic operations are supported:

```rust
let a = FE::from(10u64);
let b = FE::from(3u64);

// Addition
let sum = &a + &b;

// Subtraction
let diff = &a - &b;

// Multiplication
let product = &a * &b;

// Division
let quotient = &a / &b;

// Negation
let neg_a = -&a;

// Squaring (more efficient than a * a)
let squared = a.square();

// Exponentiation
let power = a.pow(5u64);

// Multiplicative inverse
let inverse = b.inv().unwrap();
assert_eq!(&b * &inverse, FE::one());
```

### Working with References

For performance, lambdaworks supports operations on both values and references:

```rust
let a = FE::from(5u64);
let b = FE::from(3u64);

// All of these work:
let r1 = &a + &b;    // References
let r2 = a.clone() + b.clone();  // Values
let r3 = &a + b.clone();  // Mixed
```

Using references avoids unnecessary copying, which is important for large field elements.

## Supported Prime Fields

lambdaworks includes several optimized prime field implementations:

| Field | Modulus | Use Case |
|-------|---------|----------|
| `Stark252PrimeField` | $2^{251} + 17 \cdot 2^{192} + 1$ | STARKs, Cairo |
| `Babybear31PrimeField` | $2^{31} - 2^{27} + 1$ | Fast proving |
| `Mersenne31Field` | $2^{31} - 1$ | Fast proving |
| `U64GoldilocksPrimeField` | $2^{64} - 2^{32} + 1$ | Plonky2 compatible |
| `BLS12381ScalarField` | BLS12-381 curve order | Ethereum, SNARKs |
| `BN254ScalarField` | BN254 curve order | Ethereum, SNARKs |

### FFT-Friendly Fields

Some fields are called "FFT-friendly" because their multiplicative group size is divisible by a large power of 2. This property enables efficient Fast Fourier Transform operations, which are critical for polynomial manipulation in proof systems.

For a field to be FFT-friendly, $p - 1 = 2^k \cdot m$ where $k$ is large and $m$ is odd. For example:

- Stark252: $p - 1 = 2^{192} \cdot m$
- BabyBear: $p - 1 = 2^{27} \cdot m$
- Goldilocks: $p - 1 = 2^{32} \cdot m$

## Defining Custom Fields

You can define your own prime field by implementing the `IsModulus` trait:

```rust
use lambdaworks_math::field::fields::montgomery_backed_prime_fields::{
    IsModulus, MontgomeryBackendPrimeField
};
use lambdaworks_math::unsigned_integer::element::U256;
use lambdaworks_math::field::element::FieldElement;

// Define the modulus
#[derive(Clone, Debug)]
pub struct MyFieldConfig;

impl IsModulus<U256> for MyFieldConfig {
    // Prime: 2^255 - 19 (Curve25519 field)
    const MODULUS: U256 = U256::from_hex_unchecked(
        "7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed"
    );
}

// Create the field type (4 = number of 64-bit limbs)
pub type MyField = MontgomeryBackendPrimeField<MyFieldConfig, 4>;
pub type MyFieldElement = FieldElement<MyField>;

// Use it
let a = MyFieldElement::from(42u64);
let b = a.square();
```

## Montgomery Representation

lambdaworks uses Montgomery representation for efficient modular multiplication. In this representation, a number $a$ is stored as $a \cdot R \pmod{p}$, where $R = 2^{64k}$ and $k$ is the number of limbs.

This representation allows multiplication without expensive division operations. The library handles the conversion automatically:

```rust
let a = FE::from(3u64);

// Internal representation (Montgomery form)
println!("Internal: {:?}", a);  // Shows Montgomery form

// Get the actual value
let repr = a.representative();
println!("Value: {:?}", repr);  // Shows 3

// Convert to hex string of the value
let hex = a.representative().to_hex();
```

Key functions for conversion:

| Function | Purpose |
|----------|---------|
| `from(n)` | Create from integer (converts to Montgomery) |
| `from_hex_unchecked(s)` | Create from hex (converts to Montgomery) |
| `representative()` | Get value as integer (from Montgomery) |
| `to_bytes_be()` | Get bytes of the value (big-endian) |
| `to_bytes_le()` | Get bytes of the value (little-endian) |

## Field Extensions

Some cryptographic applications require field extensions, which are larger fields built from a base field. lambdaworks supports quadratic, cubic, and higher-degree extensions.

### Quadratic Extensions

A quadratic extension $\mathbb{F}_{p^2}$ is constructed by adjoining a root of an irreducible polynomial $x^2 - \alpha$, where $\alpha$ is not a square in $\mathbb{F}_p$.

Elements are represented as $a + b \cdot i$ where $i^2 = \alpha$:

```rust
use lambdaworks_math::field::fields::fft_friendly::babybear::Babybear31PrimeField;
use lambdaworks_math::field::extensions::quadratic::QuadraticExtensionField;

// BabyBear quadratic extension (using i^2 = -1)
type Fp2 = QuadraticExtensionField<Babybear31PrimeField, Babybear31PrimeField>;
type Fp2Element = FieldElement<Fp2>;

// Create elements: a + b*i
let x = Fp2Element::new([
    FieldElement::from(3),  // Real part
    FieldElement::from(4),  // Imaginary part
]);
```

### Tower Extensions

Higher-degree extensions can be built as towers. For example, $\mathbb{F}_{p^{12}}$ for pairings is typically constructed as:

$$\mathbb{F}_p \rightarrow \mathbb{F}_{p^2} \rightarrow \mathbb{F}_{p^6} \rightarrow \mathbb{F}_{p^{12}}$$

lambdaworks provides these extensions for pairing-friendly curves like BLS12-381 and BN254.

## Batch Operations

When performing the same operation on many elements, batch operations can be more efficient:

```rust
// Batch inversion is much faster than individual inversions
let elements = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];
let mut inversions = elements.clone();
FieldElement::inplace_batch_inverse(&mut inversions).unwrap();

// Verify
for (a, a_inv) in elements.iter().zip(inversions.iter()) {
    assert_eq!(a * a_inv, FE::one());
}
```

## Performance Tips

1. **Use references** when possible to avoid copying large field elements.

2. **Use `square()`** instead of `a * a` for squaring operations.

3. **Use batch operations** for multiple inversions or multiplications.

4. **Choose FFT-friendly fields** when polynomial operations are frequent.

5. **Enable the `parallel` feature** for multi-threaded operations on vectors.

## The IsField Trait

All fields implement the `IsField` trait, which defines the core operations:

```rust
pub trait IsField: Clone {
    type BaseType;

    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType;
    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType;
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType;
    fn neg(a: &Self::BaseType) -> Self::BaseType;
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError>;
    fn zero() -> Self::BaseType;
    fn one() -> Self::BaseType;
    // ... more methods
}
```

The `IsPrimeField` trait extends this with:

```rust
pub trait IsPrimeField: IsField {
    type RepresentativeType;

    fn representative(a: &Self::BaseType) -> Self::RepresentativeType;
    fn from_hex(hex_string: &str) -> Result<Self::BaseType, CreationError>;
    // ... more methods
}
```

## Further Reading

For more details on finite field theory:

1. [Developer Math Survival Kit](https://blog.lambdaclass.com/math-survival-kit-for-developers/) - Lambda Class blog
2. [Montgomery Arithmetic from a Software Perspective](https://eprint.iacr.org/2017/1057.pdf)
3. [Efficient Software Implementation of Finite Fields](https://www.sandeep.de/my/papers/2006_ActaApplMath_EfficientSoftFiniteF.pdf)
