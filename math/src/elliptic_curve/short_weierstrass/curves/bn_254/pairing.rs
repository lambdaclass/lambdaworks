use super::{curve::BN254Curve, field_extension::Degree12ExtensionField, twist::BN254TwistCurve};
use crate::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::point::ShortWeierstrassProjectivePoint, traits::IsPairing,
    },
    field::element::FieldElement,
};

#[derive(Clone)]
pub struct BN254AtePairing;
impl IsPairing for BN254AtePairing {
    type G1Point = ShortWeierstrassProjectivePoint<BN254Curve>;
    type G2Point = ShortWeierstrassProjectivePoint<BN254TwistCurve>;
    type OutputField = Degree12ExtensionField;

    /// Compute the product of the ate pairings for a list of point pairs.
    fn compute_batch(
        pairs: &[(&Self::G1Point, &Self::G2Point)],
    ) -> FieldElement<Self::OutputField> {
        let mut result = FieldElement::one();
        for (p, q) in pairs {
            if !p.is_neutral_element() && !q.is_neutral_element() {
                let p = p.to_affine();
                let q = q.to_affine();
                result = result * miller(&q, &p);
            }
        }
        final_exponentiation(&result)
    }
}

/// This is equal to the frobenius trace of the BN 254 curve minus one.
#[allow(unused)]
const MILLER_LOOP_CONSTANT: u64 = 0xFFFFFFFFF;

#[allow(unused)]
fn double_accumulate_line(
    t: &mut ShortWeierstrassProjectivePoint<BN254TwistCurve>,
    p: &ShortWeierstrassProjectivePoint<BN254Curve>,
    accumulator: &mut FieldElement<Degree12ExtensionField>,
) {
    todo!();
    // Implementation specific to BN254 curve
}

#[allow(unused)]
fn add_accumulate_line(
    t: &mut ShortWeierstrassProjectivePoint<BN254TwistCurve>,
    q: &ShortWeierstrassProjectivePoint<BN254TwistCurve>,
    p: &ShortWeierstrassProjectivePoint<BN254Curve>,
    accumulator: &mut FieldElement<Degree12ExtensionField>,
) {
    todo!();
    // Implementation specific to BN254 curve
}

#[allow(unused)]
fn miller(
    q: &ShortWeierstrassProjectivePoint<BN254TwistCurve>,
    p: &ShortWeierstrassProjectivePoint<BN254Curve>,
) -> FieldElement<Degree12ExtensionField> {
    let mut r = q.clone();
    let mut f = FieldElement::<Degree12ExtensionField>::one();
    let mut miller_loop_constant = MILLER_LOOP_CONSTANT;
    let mut miller_loop_constant_bits: Vec<bool> = vec![];

    while miller_loop_constant > 0 {
        miller_loop_constant_bits.insert(0, (miller_loop_constant & 1) == 1);
        miller_loop_constant >>= 1;
    }

    for bit in miller_loop_constant_bits[1..].iter() {
        double_accumulate_line(&mut r, p, &mut f);
        if *bit {
            add_accumulate_line(&mut r, q, p, &mut f);
        }
    }
    f.inv().unwrap()
}

#[allow(unused)]
fn final_exponentiation(
    base: &FieldElement<Degree12ExtensionField>,
) -> FieldElement<Degree12ExtensionField> {
    /*
    const PHI_DIVIDED_BY_R: UnsignedInteger<20> = UnsignedInteger::from_hex_unchecked("f686b3d807d01c0bd38c3195c899ed3cde88eeb996ca394506632528d6a9a2f230063cf081517f68f7764c28b6f8ae5a72bce8d63cb9f827eca0ba621315b2076995003fc77a17988f8761bdc51dc2378b9039096d1b767f17fcbde783765915c97f36c6f18212ed0b283ed237db421d160aeb6a1e79983774940996754c8c71a2629b0dea236905ce937335d5b68fa9912aae208ccf1e516c3f438e3ba79");

    let f1 = base.conjugate() * base.inv().unwrap();
    let f2 = frobenius_square(&f1) * f1;
    f2.pow(PHI_DIVIDED_BY_R)
    */
    todo!();
}

//Tests

#[cfg(test)]
mod tests {
    use crate::{
        cyclic_group::IsGroup, elliptic_curve::traits::IsEllipticCurve,
        unsigned_integer::element::U384,
    };

    use super::*;

    #[test]
    fn test_double_accumulate_line_doubles_point_correctly() {
        let g1 = BN254Curve::generator();
        let g2 = BN254TwistCurve::generator();
        let mut r = g2.clone();
        let mut f = FieldElement::one();
        double_accumulate_line(&mut r, &g1, &mut f);
        assert_eq!(r, g2.operate_with(&g2));
    }

    #[test]
    fn test_add_accumulate_line_adds_points_correctly() {
        let g1 = BN254Curve::generator();
        let g = BN254TwistCurve::generator();
        let a: u64 = 12;
        let b: u64 = 23;
        let g2 = g.operate_with_self(a).to_affine();
        let g3 = g.operate_with_self(b).to_affine();
        let expected = g.operate_with_self(a + b);
        let mut r = g2;
        let mut f = FieldElement::one();
        add_accumulate_line(&mut r, &g3, &g1, &mut f);
        assert_eq!(r, expected);
    }

    #[test]
    fn batch_ate_pairing_bilinearity() {
        let p = BN254Curve::generator();
        let q = BN254TwistCurve::generator();
        let a = U384::from_u64(11);
        let b = U384::from_u64(93);

        let result = BN254AtePairing::compute_batch(&[
            (
                &p.operate_with_self(a).to_affine(),
                &q.operate_with_self(b).to_affine(),
            ),
            (
                &p.operate_with_self(a * b).to_affine(),
                &q.neg().to_affine(),
            ),
        ]);
        assert_eq!(result, FieldElement::one());
    }

    #[test]
    fn ate_pairing_returns_one_when_one_element_is_the_neutral_element() {
        let p = BN254Curve::generator().to_affine();
        let q = ShortWeierstrassProjectivePoint::neutral_element();
        let result = BN254AtePairing::compute_batch(&[(&p.to_affine(), &q)]);
        assert_eq!(result, FieldElement::one());

        let p = ShortWeierstrassProjectivePoint::neutral_element();
        let q = BN254TwistCurve::generator();
        let result = BN254AtePairing::compute_batch(&[(&p, &q.to_affine())]);
        assert_eq!(result, FieldElement::one());
    }
}
