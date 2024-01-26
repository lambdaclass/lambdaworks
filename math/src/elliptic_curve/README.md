# Elliptic curves

This folder contains the different elliptic curve models currently supported by lambdaworks. For an overview of the curve models, their addition formulas and coordinate systems, see [Hyperelliptic](https://hyperelliptic.org/EFD/g1p/index.html). The models currently supported are:
- [Short Weierstrass](https://github.com/lambdaclass/lambdaworks/tree/main/math/src/elliptic_curve/short_weierstrass)
- [Twisted Edwards](https://github.com/lambdaclass/lambdaworks/tree/main/math/src/elliptic_curve/edwards)
- [Montgomery](https://github.com/lambdaclass/lambdaworks/tree/main/math/src/elliptic_curve/montgomery)

Each of the curve models can have one or more coordinate systems, such as homogeneous projective, Jacobian, XZ coordinates, etc. These are used for reasons of performance. It is possible to define an operation, $\oplus$, taking two points over an elliptic curve, $E$ and obtain a third one, such that $(E, \oplus)$ is a group. 

## Short Weierstrass

The following curves are currently supported:
- [BLS12-377](https://github.com/lambdaclass/lambdaworks/tree/main/math/src/elliptic_curve/short_weierstrass/curves/bls12_377), a pairing-friendly elliptic curve (pairing implementation pending).
- [BLS12-381](https://github.com/lambdaclass/lambdaworks/tree/main/math/src/elliptic_curve/short_weierstrass/curves/bls12_381), a pairing-friendly elliptic curve.
- [Pallas](https://github.com/lambdaclass/lambdaworks/tree/main/math/src/elliptic_curve/short_weierstrass/curves/pallas), useful for recursive SNARKs when used with Vesta.
- [Vesta](https://github.com/lambdaclass/lambdaworks/tree/main/math/src/elliptic_curve/short_weierstrass/curves/vesta), useful for recursive SNARKs when used with Pallas.
- [Starknet's curve](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/elliptic_curve/short_weierstrass/curves/stark_curve.rs)

## Twisted Edwards

The following curves are currently supported:
- [Ed448Goldilocks](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/elliptic_curve/edwards/curves/ed448_goldilocks.rs)
- [Bandersnatch](https://github.com/lambdaclass/lambdaworks/tree/main/math/src/elliptic_curve/edwards/curves/bandersnatch)
- [TinyJubJub](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/elliptic_curve/edwards/curves/tiny_jub_jub.rs), only for learning purposes.

## Montgomery

The following curves are currently supported:
- [TinyJubJub](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/elliptic_curve/montgomery/curves/tiny_jub_jub.rs), only for learning purposes.

## Implementing Elliptic Curves in lambdaworks

In order to define your elliptic curve in lambdaworks, you need to implement some traits:
- `IsEllipticCurve`
- `IsShortWeierstrass`
- `IsEdwards`
- `IsMontgomery`

To create an elliptic curve in Short Weiestrass form, we have to implement the traits `IsEllipticCurve` and `IsShortWeierstrass` (If you want a twisted Edwards, use `IsEdwards`. For Montgomery form, use `IsMontgomery`). Below we show how the Pallas curve is defined:
```rust
#[derive(Clone, Debug)]
pub struct PallasCurve;

impl IsEllipticCurve for PallasCurve {
    type BaseField = Pallas255PrimeField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        Self::PointRepresentation::new([
            -FieldElement::<Self::BaseField>::one(),
            FieldElement::<Self::BaseField>::from(2),
            FieldElement::one(),
        ])
    }
}

impl IsShortWeierstrass for PallasCurve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(5)
    }
}
```

Here, $a$ and $b$ are the parameters for the Elliptic Curve in Weiestrass form. All curve models have their `defining_equation` method, which allows us to check whether a given $(x,y)$ belongs to the elliptic curve. The `BaseField` is where the coordinates $x,y$ of the curve live. `generator()` gives a point $P$, such that, by doing $P, 2P, 3P, ... , nP$ ($2 P = P \oplus P$) we span all the elements that belong to the Elliptic Curve.

To implement the `IsShortWeierstrass`, you need to first implement `IsEllipticCurve`. It has 3 methods, two of which need to be implemented for each curve `fn a()` and `fn b()` (the curve parameters) and `fn defining_equation()`, which computes $y^2 - x^3 - a x - b$. If this result is equal to the base field element 0, then the point satisfies the curve equation and is valid (note, however, that the point may not be in the appropriate subgroup of the curve!)

## Defining points and operating with the curves

All curves implement the trait `FromAffine`, which lets us define points by providing the pair of values $(x,y)$, where $x$ and $y$ should be in the `BaseField` of the curve. For example
```rust
let x = FE::from_hex_unchecked(
            "bd1e740e6b1615ae4c508148ca0c53dbd43f7b2e206195ab638d7f45d51d6b5",
        );
let y = FE::from_hex_unchecked(
            "13aacd107ca10b7f8aab570da1183b91d7d86dd723eaa2306b0ef9c5355b91d8",
        );
PallasCurve::create_point_from_affine(x, y).unwrap()
```
The function has to check whether the point is valid, and, if not, returns an error.

Each form and coordinate model has to implement the `IsGroup` trait, which will give us all the necessary operations for the group. We need to provide expressions for:
- `fn neutral_element()`, the neutral element for the group operation. In the case of elliptic curves, this is the point at infinity.
- `fn operate_with`, which defines the group operation; it takes two elements in the group and outputs a third one.
- `fn neg`, which gives the inverse of the element.
It also provides the method `fn operate_with_self`, which is used to indicate that repeteadly add one element against itself $n$ times. Here, $n$ should implement the `IsUnsignedInteger` trait. In the case of elliptic curves, this provides the scalar multiplication, $n P$, based on the double and add algorithm (square and multiply).

Operating is done in the following way:
```rust
// We get a point
let g = PallasCurve::generator();
let g2 = g.operate_with_self(2_u16);
let g3 = g.operate_with_other(&g2);
```
`operate_with_self` takes as argument anything that implements the `IsUnsignedInteger` trait. `operate_with_other` takes as argument another point in the elliptic curve. When we operate this way, the $z$ coordinate in the result may be different from $1$. We can transform it back to affine form by using `to_affine`. For example,
```rust
let g = BLS12381Curve::generator();
let g2 = g.operate_with_self(2_u64);

// get x and y from affine coordinates
let g2_affine = g2.to_affine();
let x = g2_affine.x();
let y = g2_affine.y();
```

## Multiscalar multiplication

One common operation for different proof systems is the Mutiscalar Multiplication (MSM), which is given by a set of points $P_0 , P_1 , P_2 , ... , P_n$ and scalars $a_0 , a_1 , a_2 ... n_n$ (the scalars belong to the scalar field of the elliptic curve, which is the field whose size matches the size of the elliptic curve's group):
$$R = \sum_k a_k P_k$$ 
This operation could be implemented by using `operate_with_self` with each point and scalar and then add the results using `operate_with`, but this is not efficient. lambdaworks provides an optimized [MSM using Pippenger's algorithm](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/msm/pippenger.rs). A naïve version is given [here](https://github.com/lambdaclass/lambdaworks/blob/main/math/src/msm/naive.rs). Below we show how to use MSM in the context of a polynomial commitment scheme: the scalars are the coefficients of the polynomials and the points are provided by an SRS.
```rust
fn commit(&self, p: &Polynomial<FieldElement<F>>) -> Self::Commitment {
        let coefficients: Vec<_> = p
            .coefficients
            .iter()
            .map(|coefficient| coefficient.representative())
            .collect();
        msm(
            &coefficients,
            &self.srs.powers_main_group[..coefficients.len()],
        )
        .expect("`points` is sliced by `cs`'s length")
    }
```
## Pairing-friendly elliptic curves

Pairings are an important calculation for BLS signatures and the KZG polynomial commitment scheme. These are functions mapping elements from groups of order $r$ belonging to an elliptic curve to the set of $r$-th roots of unity, $e: G_1 \times G_2 \rightarrow G_t$. They satisfy two properties:
1. Bilinearity
2. Non-degeneracy
Not all elliptic curves have efficiently computable pairings. If the curve is pairing-friendly, we can implement the trait `IsPairing`. Examples of pairing-friendly curves are BLS12-381, BLS12-377, BN254. Curves such as Pallas, Vesta, secp256k1 are not pairing-friendly.
