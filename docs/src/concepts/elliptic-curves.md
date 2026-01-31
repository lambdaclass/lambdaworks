# Elliptic Curves

Elliptic curves are fundamental to modern cryptography, providing the mathematical structure for digital signatures, key exchange, and zero-knowledge proofs. This document explains elliptic curve concepts and their implementation in lambdaworks.

## What is an Elliptic Curve?

An elliptic curve over a finite field is the set of points $(x, y)$ satisfying a specific equation, plus a special "point at infinity." The most common form is the Short Weierstrass equation:

$$y^2 = x^3 + ax + b$$

where $a$ and $b$ are constants from the underlying field, and the discriminant $4a^3 + 27b^2 \neq 0$ (to ensure no singular points).

Elliptic curves have a remarkable property: you can define an addition operation on points that makes them form a mathematical group. This group structure is what makes elliptic curves useful for cryptography.

## Curve Forms in lambdaworks

lambdaworks supports three elliptic curve forms:

### Short Weierstrass

The most common form, used by most cryptographic curves:

$$y^2 = x^3 + ax + b$$

Supported curves: BLS12-381, BLS12-377, BN254, secp256k1, Pallas, Vesta

### Twisted Edwards

An alternative form with faster addition formulas:

$$ax^2 + y^2 = 1 + dx^2y^2$$

Supported curves: Ed448-Goldilocks, Bandersnatch

### Montgomery

Efficient for scalar multiplication:

$$by^2 = x^3 + ax^2 + x$$

Supported curves: Curve25519 (via TinyJubJub for learning)

## Working with Curves

### Getting the Generator

Every elliptic curve has a generator point $G$ that, when repeatedly added to itself, generates all points in the curve's group:

```rust
use lambdaworks_math::elliptic_curve::{
    short_weierstrass::curves::bls12_381::curve::BLS12381Curve,
    traits::IsEllipticCurve,
};

// Get the generator point
let g = BLS12381Curve::generator();
```

### Creating Points from Coordinates

You can create a point from its affine coordinates $(x, y)$:

```rust
use lambdaworks_math::elliptic_curve::{
    short_weierstrass::curves::pallas::curve::PallasCurve,
    traits::IsEllipticCurve,
};
use lambdaworks_math::field::element::FieldElement;

type FE = FieldElement<PallasCurve::BaseField>;

let x = FE::from_hex_unchecked("...");
let y = FE::from_hex_unchecked("...");

// This validates that (x, y) is on the curve
let point = PallasCurve::create_point_from_affine(x, y).unwrap();
```

### Point Operations

The group operation on elliptic curves is called "addition." lambdaworks provides several ways to combine points:

```rust
use lambdaworks_math::cyclic_group::IsGroup;

let g = BLS12381Curve::generator();

// Point addition: P + Q
let g2 = g.operate_with(&g);  // 2G

// Scalar multiplication: n * P
let g5 = g.operate_with_self(5u64);  // 5G

// Combine addition and scalar multiplication
let g7 = g5.operate_with(&g2);  // 7G

// Verify
assert_eq!(g7, g.operate_with_self(7u64));

// Get the neutral element (point at infinity)
let identity = BLS12381Curve::neutral_element();
assert_eq!(g.operate_with(&identity), g);
```

### Coordinate Systems

lambdaworks internally uses projective coordinates for efficiency. A point $(x, y)$ in affine coordinates becomes $(X : Y : Z)$ in projective coordinates, where:

$$x = X/Z, \quad y = Y/Z$$

This representation avoids expensive field inversions during point addition. To convert back:

```rust
let g = BLS12381Curve::generator();
let g5 = g.operate_with_self(5u64);

// Convert to affine coordinates
let affine = g5.to_affine();
let x = affine.x();
let y = affine.y();
```

## Supported Curves

### Pairing-Friendly Curves

These curves support bilinear pairings, essential for SNARKs and BLS signatures:

| Curve | Field Size | Use Case |
|-------|------------|----------|
| BLS12-381 | 381 bits | Ethereum 2.0, Zcash |
| BLS12-377 | 377 bits | Aleo |
| BN254 | 254 bits | Ethereum (precompiles) |

### Cycle Curves

Pairs of curves where each curve's scalar field equals the other's base field:

| Pair | Use Case |
|------|----------|
| Pallas/Vesta | Recursive SNARKs |
| Grumpkin/BN254 | Noir circuits |

### ECDSA Curves

Standard curves for digital signatures:

| Curve | Standard |
|-------|----------|
| secp256k1 | Bitcoin, Ethereum |
| secp256r1 (P-256) | NIST, TLS |

### Edwards Curves

Fast signature verification:

| Curve | Use Case |
|-------|----------|
| Ed448-Goldilocks | High security |
| Bandersnatch | Ethereum research |

## Multi-Scalar Multiplication (MSM)

MSM computes $\sum_i s_i \cdot P_i$ where $s_i$ are scalars and $P_i$ are points. This is a critical operation for proof systems:

```rust
use lambdaworks_math::msm::pippenger::msm;

// Given points P0, P1, P2 and scalars s0, s1, s2
// Compute s0*P0 + s1*P1 + s2*P2
let points = vec![p0, p1, p2];
let scalars = vec![s0.representative(), s1.representative(), s2.representative()];

let result = msm(&scalars, &points).unwrap();
```

lambdaworks uses Pippenger's algorithm for efficient MSM, which is much faster than computing each scalar multiplication separately.

## Pairings

Pairing-friendly curves support a bilinear map:

$$e: G_1 \times G_2 \rightarrow G_T$$

where $G_1$ is the curve over the base field, $G_2$ is the curve over an extension field, and $G_T$ is a multiplicative group in a larger extension field.

The key property is bilinearity:

$$e(aP, bQ) = e(P, Q)^{ab}$$

This property enables KZG commitments and Groth16 proofs:

```rust
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bn_254::{
    curve::BN254Curve,
    twist::BN254TwistCurve,
    pairing::BN254AtePairing,
};
use lambdaworks_math::elliptic_curve::traits::IsPairing;

let p = BN254Curve::generator();
let q = BN254TwistCurve::generator();

// Compute the pairing
let result = BN254AtePairing::compute_batch(&[(&p, &q)]).unwrap();

// Verify bilinearity: e(2P, 3Q) = e(P, Q)^6
let p2 = p.operate_with_self(2u64);
let q3 = q.operate_with_self(3u64);
let left = BN254AtePairing::compute_batch(&[(&p2, &q3)]).unwrap();

let base = BN254AtePairing::compute_batch(&[(&p, &q)]).unwrap();
let right = base.pow(6u64);

assert_eq!(left, right);
```

## Point Compression

Elliptic curve points can be compressed to save space. Since $y^2 = x^3 + ax + b$, given $x$ we can compute $y$ up to sign. We only need to store $x$ and one bit indicating which square root to use:

```rust
// Compression: 48 bytes -> 49 bytes for BLS12-381 G1
// (48 bytes for x, 1 bit for y sign, typically stored in spare bits)

// Get compressed representation
let compressed = point.to_compressed();

// Decompress
let decompressed = Point::from_compressed(&compressed).unwrap();
```

## Defining Custom Curves

To add a new curve, implement the appropriate traits:

```rust
use lambdaworks_math::elliptic_curve::{
    traits::IsEllipticCurve,
    short_weierstrass::traits::IsShortWeierstrass,
    short_weierstrass::point::ShortWeierstrassProjectivePoint,
};

#[derive(Clone, Debug)]
pub struct MyCurve;

impl IsEllipticCurve for MyCurve {
    type BaseField = MyBaseField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            FieldElement::from_hex_unchecked("..."),  // x
            FieldElement::from_hex_unchecked("..."),  // y
            FieldElement::one(),                       // z
        ])
    }
}

impl IsShortWeierstrass for MyCurve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)  // a = 0 for most curves
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(7)  // b = 7 for secp256k1
    }
}
```

## Security Considerations

1. **Constant-time operations**: The lambdaworks secp256k1 implementation is not constant-time and should not be used for signing. Use a dedicated library for production ECDSA.

2. **Subgroup checks**: When accepting points from external sources, verify they are in the correct subgroup to prevent small subgroup attacks.

3. **Cofactor**: Some curves have a cofactor $h > 1$. The full curve group has order $h \cdot r$ where $r$ is the prime subgroup order.

## Further Reading

1. [What Every Developer Needs to Know About Elliptic Curves](https://blog.lambdaclass.com/what-every-developer-needs-to-know-about-elliptic-curves/)
2. [BLS12-381 For The Rest Of Us](https://hackmd.io/@benjaminion/bls12-381)
3. [BN254 For The Rest Of Us](https://hackmd.io/@jpw/bn254)
4. [How We Implemented BN254 Ate Pairing](https://blog.lambdaclass.com/how-we-implemented-the-bn254-ate-pairing-in-lambdaworks/)
5. [HyperElliptic EFD](https://hyperelliptic.org/EFD/) - Formulas for elliptic curve operations
