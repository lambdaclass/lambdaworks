# Migration Guides

This page helps you migrate to lambdaworks from other cryptographic libraries.

## From Arkworks

Arkworks and lambdaworks share similar concepts but have different APIs.

### Field Elements

**Arkworks:**
```rust
use ark_ff::{Field, PrimeField};
use ark_bls12_381::Fr;

let a = Fr::from(42u64);
let b = Fr::from(7u64);
let c = a * b;
let inv = b.inverse().unwrap();
```

**Lambdaworks:**
```rust
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrField;

type Fr = FieldElement<FrField>;

let a = Fr::from(42u64);
let b = Fr::from(7u64);
let c = &a * &b;  // Note: use references for efficiency
let inv = b.inv().unwrap();
```

**Key differences:**
- Lambdaworks uses `FieldElement<F>` wrapper vs Arkworks' direct field types
- Prefer references (`&a * &b`) in lambdaworks to avoid cloning
- `inverse()` → `inv()`
- `square()` is the same in both

### Elliptic Curves

**Arkworks:**
```rust
use ark_bls12_381::{G1Projective, G1Affine};
use ark_ec::Group;

let g = G1Projective::generator();
let result = g * Fr::from(5u64);
let affine = result.into_affine();
```

**Lambdaworks:**
```rust
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use lambdaworks_math::cyclic_group::IsGroup;

let g = BLS12381Curve::generator();
let result = g.operate_with_self(5u64);  // Scalar multiplication
let affine = result.to_affine();
```

**Key differences:**
- Scalar multiplication: `g * scalar` → `g.operate_with_self(scalar)`
- Point addition: `a + b` → `a.operate_with(&b)`
- `into_affine()` → `to_affine()`

### Polynomials

**Arkworks:**
```rust
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial};

let poly = DensePolynomial::from_coefficients_vec(vec![Fr::from(1), Fr::from(2)]);
let eval = poly.evaluate(&Fr::from(5));
```

**Lambdaworks:**
```rust
use lambdaworks_math::polynomial::Polynomial;

let poly = Polynomial::new(&[Fr::from(1), Fr::from(2)]);
let eval = poly.evaluate(&Fr::from(5));
```

### Pairing Operations

**Arkworks:**
```rust
use ark_bls12_381::{Bls12_381, G1Projective, G2Projective};
use ark_ec::pairing::Pairing;

let result = Bls12_381::pairing(g1, g2);
```

**Lambdaworks:**
```rust
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::pairing::BLS12381AtePairing;
use lambdaworks_math::elliptic_curve::traits::IsPairing;

let result = BLS12381AtePairing::compute_batch(&[(g1, g2)]);
```

## From Halo2

Halo2 is more focused on circuit development. Lambdaworks provides lower-level primitives.

### Field Arithmetic

**Halo2:**
```rust
use halo2_proofs::pasta::Fp;

let a = Fp::from(42u64);
let b = a.square();
let c = a.invert().unwrap();
```

**Lambdaworks:**
```rust
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::pallas_field::PallasField;

type Fp = FieldElement<PallasField>;

let a = Fp::from(42u64);
let b = a.square();
let c = a.inv().unwrap();
```

### Curves

Both Pallas and Vesta curves are available in lambdaworks:

```rust
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::pallas::curve::PallasCurve;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::vesta::curve::VestaCurve;
```

## Common Patterns

### Batch Operations

**Batch Inversion** (much faster than individual inversions):
```rust
// Instead of:
let invs: Vec<_> = elements.iter().map(|e| e.inv().unwrap()).collect();

// Use:
let mut elements_clone = elements.clone();
FieldElement::inplace_batch_inverse(&mut elements_clone).unwrap();
```

### Working with Bytes

**Serialization:**
```rust
// To bytes
let bytes_be = element.to_bytes_be();
let bytes_le = element.to_bytes_le();

// From bytes
let elem = FieldElement::<F>::from_bytes_be(&bytes)?;
let elem = FieldElement::<F>::from_bytes_le(&bytes)?;
```

### Hex Representation

```rust
// Create from hex
let elem = FieldElement::<F>::from_hex("0x1234abcd")?;
let elem = FieldElement::<F>::from_hex_unchecked("0x1234abcd"); // panics on invalid

// To hex
let hex = elem.to_hex();
```

## Feature Comparison

| Feature | Arkworks | Halo2 | Lambdaworks |
|---------|----------|-------|-------------|
| BLS12-381 | Yes | No | Yes |
| BN254 | Yes | Yes | Yes |
| Pallas/Vesta | Yes | Yes | Yes |
| STARK fields | No | No | Yes |
| BabyBear/Mersenne31 | No | No | Yes |
| GPU acceleration | Limited | No | Yes (Metal) |
| WebAssembly | Yes | Yes | Yes |
| no_std | Yes | Partial | Yes |

## Performance Notes

1. **Montgomery representation**: Both Arkworks and lambdaworks use Montgomery form internally. Conversion happens automatically.

2. **Assembly optimizations**: Lambdaworks has optimized assembly for x86_64 and ARM64.

3. **Parallel operations**: Enable with `features = ["parallel"]` for multi-threaded FFT and MSM.

4. **GPU acceleration**: Lambdaworks supports Metal on macOS with `features = ["metal"]`.

## Getting Help

- [Telegram](https://t.me/lambdaworks) - Community chat
- [GitHub Issues](https://github.com/lambdaclass/lambdaworks/issues) - Bug reports and feature requests
- [Examples](https://github.com/lambdaclass/lambdaworks/tree/main/examples) - Working code samples
