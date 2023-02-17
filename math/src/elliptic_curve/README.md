
```rust
fn do_something_with_an_elliptic_curve<T: EllipticCurve + Pairing>(curve: T) {
    let g1 = T::subgroup_generator(); // EllipticCurve trait
    let g2 = T::secondary_subgroup_generator(); // Pairing trait
    T::pairing(g1, g2); // Pairing trait

    let p1 = T::create_point_from_affine(12, 12); // EllipticCurve trait

    let s = g1.operate_with(g2); // IsGroup trait
}
```
