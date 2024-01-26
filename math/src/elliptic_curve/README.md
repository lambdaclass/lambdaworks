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

```rust
fn do_something_with_an_elliptic_curve<T: EllipticCurve + Pairing>(curve: T) {
    let g1 = T::subgroup_generator(); // EllipticCurve trait
    let g2 = T::secondary_subgroup_generator(); // Pairing trait
    T::pairing(g1, g2); // Pairing trait

    let p1 = T::create_point_from_affine(12, 12); // EllipticCurve trait

    let s = g1.operate_with(g2); // IsGroup trait
}
```
