# Elliptic curves

This folder contains the different elliptic curve models currently supported by lambdaworks. For an overview of the curve models, their addition formulas and coordinate systems, see [Hyperelliptic](https://hyperelliptic.org/EFD/g1p/index.html). The models currently supported are:
- [Short Weierstrass](https://github.com/lambdaclass/lambdaworks/tree/main/math/src/elliptic_curve/short_weierstrass)
- [Twisted Edwards](https://github.com/lambdaclass/lambdaworks/tree/main/math/src/elliptic_curve/edwards)
- [Montgomery](https://github.com/lambdaclass/lambdaworks/tree/main/math/src/elliptic_curve/montgomery)

Each of the curve models can have one or more coordinate systems, such as homogeneous projective, Jacobian, XZ coordinates, etc. These are used for reasons of performance. It is possible to define an operation, $\oplus$, taking two points over an elliptic curve, $E$ and obtain a third one, such that $(E, \oplus)$ is a group. 

```rust
fn do_something_with_an_elliptic_curve<T: EllipticCurve + Pairing>(curve: T) {
    let g1 = T::subgroup_generator(); // EllipticCurve trait
    let g2 = T::secondary_subgroup_generator(); // Pairing trait
    T::pairing(g1, g2); // Pairing trait

    let p1 = T::create_point_from_affine(12, 12); // EllipticCurve trait

    let s = g1.operate_with(g2); // IsGroup trait
}
```
