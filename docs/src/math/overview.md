# Math Library Overview

The `lambdaworks-math` crate provides the core mathematical primitives needed for cryptographic protocols and proof systems.

## Core Concepts

### Field Elements

All arithmetic in lambdaworks happens over finite fields. The `FieldElement<F>` type wraps field operations:

```rust
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

type FE = FieldElement<Stark252PrimeField>;

let a = FE::from(42u64);
let b = FE::from(7u64);

// Basic operations
let sum = &a + &b;
let product = &a * &b;
let quotient = &a / &b;
let squared = a.square();
let inverse = b.inv().unwrap();
let power = a.pow(100u64);
```

### Available Fields

**FFT-Friendly Fields** (optimized for polynomial operations):
- `Stark252PrimeField` - Used by StarkNet and STARK provers
- `Babybear31PrimeField` - 31-bit field ($2^{31} - 2^{27} + 1$)
- `Mersenne31Field` - 31-bit Mersenne prime ($2^{31} - 1$)
- `U64GoldilocksPrimeField` - 64-bit Goldilocks ($2^{64} - 2^{32} + 1$)

**Curve Fields**:
- BLS12-381, BLS12-377, BN254 (base and scalar fields)
- secp256k1, secp256r1 (ECDSA curves)
- Pallas, Vesta (cycle curves)

### Field Extensions

For pairing operations and certain proof systems, you need field extensions:

```rust
use lambdaworks_math::field::fields::fft_friendly::babybear::Babybear31PrimeField;
use lambdaworks_math::field::extensions::quadratic::QuadraticExtensionField;

// Quadratic extension of BabyBear
type Fp2 = QuadraticExtensionField<Babybear31PrimeField, Babybear31PrimeField>;
```

## Elliptic Curves

### Point Operations

```rust
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use lambdaworks_math::cyclic_group::IsGroup;

// Get generator point
let g = BLS12381Curve::generator();

// Scalar multiplication
let five_g = g.operate_with_self(5u64);

// Point addition
let six_g = g.operate_with(&five_g);
```

### Curve Types

| Type | Equation | Curves |
|------|----------|--------|
| Short Weierstrass | $y^2 = x^3 + ax + b$ | BLS12-381, BN254, secp256k1 |
| Twisted Edwards | $ax^2 + y^2 = 1 + dx^2y^2$ | Ed25519, Ed448 |
| Montgomery | $By^2 = x^3 + Ax^2 + x$ | Curve25519 |

## Polynomials

### Dense Polynomials

```rust
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

type F = U64PrimeField<65537>;
type FE = FieldElement<F>;

// Create polynomial: 1 + 2x + 3x²
let coeffs = vec![FE::from(1), FE::from(2), FE::from(3)];
let p = Polynomial::new(&coeffs);

// Evaluate at x = 5
let result = p.evaluate(&FE::from(5));

// Polynomial arithmetic
let q = Polynomial::new(&[FE::from(1), FE::from(1)]); // 1 + x
let product = &p * &q;
let sum = &p + &q;
```

### Lagrange Interpolation

```rust
// Given points (x_i, y_i), find the unique polynomial passing through them
let xs = vec![FE::from(1), FE::from(2), FE::from(3)];
let ys = vec![FE::from(1), FE::from(4), FE::from(9)]; // y = x²
let poly = Polynomial::interpolate(&xs, &ys).unwrap();
```

## Fast Fourier Transform

FFT is used for efficient polynomial multiplication and evaluation:

```rust
use lambdaworks_math::fft::cpu::roots_of_unity::get_powers_of_primitive_root;
use lambdaworks_math::fft::polynomial::FFTPoly;

// Get roots of unity for FFT of size 2^n
let order = 8u64;
let roots = get_powers_of_primitive_root::<Stark252PrimeField>(order, order as usize).unwrap();

// Efficient polynomial evaluation at all roots of unity
let poly = Polynomial::new(&coefficients);
let evaluations = poly.evaluate_fft().unwrap();
```

## Multi-Scalar Multiplication (MSM)

For computing $\sum_i s_i \cdot P_i$ efficiently:

```rust
use lambdaworks_math::msm::pippenger::msm;

// Points and scalars
let points: Vec<_> = (0..100).map(|i| g.operate_with_self(i as u64)).collect();
let scalars: Vec<_> = (0..100).map(|i| FE::from(i as u64)).collect();

// Compute MSM
let result = msm(&scalars, &points);
```

## Performance Tips

1. **Use references for arithmetic**: `&a + &b` avoids cloning
2. **Batch inversions**: Use `FieldElement::inplace_batch_inverse()` instead of individual inversions
3. **Prefer `square()` over `* self`**: Squaring is optimized
4. **Enable parallel feature**: `features = ["parallel"]` for multi-threaded operations
5. **Use FFT for polynomial multiplication**: Much faster for large polynomials

## Next Steps

- [Finite Fields Deep Dive](./fields.md) - Montgomery arithmetic, extensions
- [Elliptic Curves Guide](./curves.md) - Curve operations, pairings
- [Polynomial Operations](./polynomials.md) - FFT, interpolation, division
