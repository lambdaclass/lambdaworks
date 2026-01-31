# lambdaworks-math

The `lambdaworks-math` crate provides the mathematical foundation for all cryptographic operations in lambdaworks. It includes finite field arithmetic, elliptic curve operations, polynomial manipulation, FFT, and multi-scalar multiplication.

## Installation

```toml
[dependencies]
lambdaworks-math = "0.13.0"
```

For `no_std` environments:

```toml
[dependencies]
lambdaworks-math = { version = "0.13.0", default-features = false, features = ["alloc"] }
```

## Module Overview

| Module | Description |
|--------|-------------|
| `field` | Finite field types and arithmetic |
| `elliptic_curve` | Elliptic curve points and operations |
| `polynomial` | Univariate and multivariate polynomials |
| `fft` | Fast Fourier Transform |
| `msm` | Multi-scalar multiplication |
| `unsigned_integer` | Big integer arithmetic |
| `cyclic_group` | Group operation traits |

## Finite Fields

### Basic Usage

```rust
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

type FE = FieldElement<Stark252PrimeField>;

// Create field elements
let a = FE::from(42u64);
let b = FE::from_hex_unchecked("0x2a");
let zero = FE::zero();
let one = FE::one();

// Arithmetic
let sum = &a + &b;
let product = &a * &b;
let inverse = a.inv().expect("non-zero element");
let squared = a.square();
let power = a.pow(10u64);
```

### Available Fields

**FFT-Friendly Fields** (optimal for polynomial operations):

```rust
use lambdaworks_math::field::fields::fft_friendly::{
    stark_252_prime_field::Stark252PrimeField,
    babybear::Babybear31PrimeField,
    u64_goldilocks::U64GoldilocksPrimeField,
};
```

**Curve Scalar Fields** (for elliptic curve operations):

```rust
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::{
    bls12_381::default_types::FrField as BLS12381Fr,
    bn_254::default_types::FrField as BN254Fr,
};
```

### Custom Fields

```rust
use lambdaworks_math::field::fields::montgomery_backed_prime_fields::{
    IsModulus, MontgomeryBackendPrimeField
};
use lambdaworks_math::unsigned_integer::element::U256;

#[derive(Clone, Debug)]
pub struct MyModulus;

impl IsModulus<U256> for MyModulus {
    const MODULUS: U256 = U256::from_hex_unchecked(
        "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001"
    );
}

pub type MyField = MontgomeryBackendPrimeField<MyModulus, 4>;
```

### Field Extensions

```rust
use lambdaworks_math::field::extensions::quadratic::QuadraticExtensionField;

// Quadratic extension (Fp2)
type Fp2 = QuadraticExtensionField<BaseField, BaseField>;
```

## Elliptic Curves

### Point Operations

```rust
use lambdaworks_math::elliptic_curve::{
    short_weierstrass::curves::bls12_381::curve::BLS12381Curve,
    traits::IsEllipticCurve,
};
use lambdaworks_math::cyclic_group::IsGroup;

// Generator point
let g = BLS12381Curve::generator();

// Scalar multiplication
let g5 = g.operate_with_self(5u64);

// Point addition
let g6 = g.operate_with(&g5);

// Convert to affine
let affine = g6.to_affine();
let (x, y) = (affine.x(), affine.y());
```

### Supported Curves

**Short Weierstrass**:
```rust
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::{
    bls12_381::curve::BLS12381Curve,
    bls12_377::curve::BLS12377Curve,
    bn_254::curve::BN254Curve,
    pallas::curve::PallasCurve,
    vesta::curve::VestaCurve,
    secp256k1::curve::Secp256k1Curve,
};
```

**Twisted Edwards**:
```rust
use lambdaworks_math::elliptic_curve::edwards::curves::{
    ed448_goldilocks::Ed448GoldilocksCurve,
    bandersnatch::BandersnatchCurve,
};
```

### Pairings

```rust
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::{
    curve::BLS12381Curve,
    twist::BLS12381TwistCurve,
    pairing::BLS12381AtePairing,
};
use lambdaworks_math::elliptic_curve::traits::IsPairing;

let p = BLS12381Curve::generator();
let q = BLS12381TwistCurve::generator();

let result = BLS12381AtePairing::compute_batch(&[(&p, &q)])
    .expect("pairing computation");
```

## Polynomials

### Univariate Polynomials

```rust
use lambdaworks_math::polynomial::Polynomial;

// Create polynomial: 1 + 2x + 3x^2
let p = Polynomial::new(&[FE::from(1), FE::from(2), FE::from(3)]);

// Evaluate
let y = p.evaluate(&FE::from(5));

// Interpolate
let xs = vec![FE::from(0), FE::from(1), FE::from(2)];
let ys = vec![FE::from(1), FE::from(4), FE::from(9)];
let q = Polynomial::interpolate(&xs, &ys).expect("interpolation");

// Arithmetic
let sum = &p + &q;
let product = &p * &q;
let (quotient, remainder) = p.div_with_ref(&q);

// Degree and properties
let deg = p.degree();
let lead = p.leading_coefficient();
```

### Multilinear Polynomials

```rust
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;

// Create from evaluations over Boolean hypercube
let evals = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];
let mle = DenseMultilinearPolynomial::new(evals);

// Evaluate at arbitrary point
let point = vec![FE::from(2), FE::from(3)];
let result = mle.evaluate(point).expect("evaluation");

// Convert to univariate (for sumcheck)
let univar = mle.to_univariate();
```

## FFT

### Basic FFT

```rust
use lambdaworks_math::fft::polynomial::{evaluate_fft, interpolate_fft, multiply_fft};

// FFT evaluation (coefficients -> evaluations at roots of unity)
let evaluations = evaluate_fft(&polynomial).expect("FFT");

// Inverse FFT (evaluations -> coefficients)
let coefficients = interpolate_fft(&evaluations).expect("IFFT");

// FFT-based multiplication
let product = multiply_fft(&p1, &p2).expect("FFT multiply");
```

### Roots of Unity

```rust
use lambdaworks_math::fft::cpu::roots_of_unity::{
    get_powers_of_primitive_root,
    get_primitive_root_of_unity,
};

// Get n-th root of unity
let omega = get_primitive_root_of_unity::<Stark252PrimeField>(8)
    .expect("root exists");

// Get all powers
let roots = get_powers_of_primitive_root::<Stark252PrimeField>(8, 8)
    .expect("roots");
```

## Multi-Scalar Multiplication

```rust
use lambdaworks_math::msm::pippenger::msm;

let points = vec![p0, p1, p2, p3];
let scalars = vec![
    s0.representative(),
    s1.representative(),
    s2.representative(),
    s3.representative(),
];

// Compute s0*P0 + s1*P1 + s2*P2 + s3*P3
let result = msm(&scalars, &points).expect("MSM");
```

## Big Integers

```rust
use lambdaworks_math::unsigned_integer::element::{U256, U384, U128};

let a = U256::from_hex_unchecked("0x1234567890abcdef");
let b = U256::from(12345u64);

// Arithmetic
let sum = a + b;
let product = a * b;
let (quotient, remainder) = a.div_rem(&b);
```

## Traits Reference

### IsField

```rust
pub trait IsField: Clone {
    type BaseType;

    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType;
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType;
    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType;
    fn neg(a: &Self::BaseType) -> Self::BaseType;
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError>;
    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType;
    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool;
    fn zero() -> Self::BaseType;
    fn one() -> Self::BaseType;
    fn from_u64(x: u64) -> Self::BaseType;
    fn from_base_type(x: Self::BaseType) -> Self::BaseType;
}
```

### IsEllipticCurve

```rust
pub trait IsEllipticCurve: Clone {
    type BaseField: IsField;
    type PointRepresentation: IsGroup;

    fn generator() -> Self::PointRepresentation;
}
```

### IsGroup

```rust
pub trait IsGroup: Clone + PartialEq + Eq {
    fn neutral_element() -> Self;
    fn operate_with(&self, other: &Self) -> Self;
    fn neg(&self) -> Self;
    fn operate_with_self(&self, scalar: impl Into<u128>) -> Self;
}
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `std` | Standard library (default) |
| `alloc` | Heap allocation without std |
| `parallel` | Parallel processing with rayon |
| `cuda` | CUDA GPU acceleration |
| `asm` | Assembly optimizations |
| `lambdaworks-serde-binary` | Binary serialization |
| `lambdaworks-serde-string` | JSON serialization |

## Performance Tips

1. **Use references** to avoid copying large field elements.
2. **Use `square()`** instead of `a * a`.
3. **Use batch inversion** for multiple inversions.
4. **Enable `parallel`** for FFT and MSM on large inputs.
5. **Choose FFT-friendly fields** when polynomial operations dominate.

## Examples

See the [examples directory](https://github.com/lambdaclass/lambdaworks/tree/main/examples) for complete working examples:

1. [Shamir Secret Sharing](https://github.com/lambdaclass/lambdaworks/tree/main/examples/shamir_secret_sharing) - Polynomial interpolation
2. [Schnorr Signatures](https://github.com/lambdaclass/lambdaworks/tree/main/examples/schnorr-signature) - Elliptic curve operations
3. [RSA](https://github.com/lambdaclass/lambdaworks/tree/main/examples/rsa) - Big integer arithmetic
