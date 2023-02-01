use crate::{
    cyclic_group::IsCyclicBilinearGroup,
    elliptic_curve::{element::EllipticCurveElement, traits::HasEllipticCurveOperations},
    field::{
        element::FieldElement,
        extensions::{
            cubic::{CubicExtensionField, HasCubicNonResidue},
            quadratic::{HasQuadraticNonResidue, QuadraticExtensionField},
        },
        fields::u384_prime_field::{HasU384Constant, U384PrimeField},
    },
};
use crypto_bigint::U384;

/// Order of the base field (e.g.: order of the coordinates)
const fn order_p() -> U384 {
    U384::from_be_hex("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab")
}

/// Order of the subgroup of the curve.
const fn order_r() -> U384 {
    U384::from_be_hex("0000000000000000000000000000000073eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001")
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ModP;
impl HasU384Constant for ModP {
    const VALUE: U384 = order_p();
}

#[derive(Debug, Clone)]
pub struct QNR1;
impl HasQuadraticNonResidue for QNR1 {
    type BaseField = U384PrimeField<ModP>;

    fn residue() -> FieldElement<U384PrimeField<ModP>> {
        -FieldElement::one()
    }
}

type TF1 = QuadraticExtensionField<QNR1>;

#[derive(Debug, Clone)]
pub struct CNR;
impl HasCubicNonResidue for CNR {
    type BaseField = TF1;

    fn residue() -> FieldElement<TF1> {
        FieldElement::new([FieldElement::from(1), FieldElement::from(1)])
    }
}

type TF2 = CubicExtensionField<CNR>;

#[derive(Debug, Clone)]
pub struct QNR2;
impl HasQuadraticNonResidue for QNR2 {
    type BaseField = TF2;

    fn residue() -> FieldElement<TF2> {
        FieldElement::new([
            FieldElement::zero(),
            FieldElement::one(),
            FieldElement::zero(),
        ])
    }
}

type TF3 = QuadraticExtensionField<QNR2>;

/// The description of the curve.
#[derive(Clone, Debug)]
pub struct BLS12381Curve;
impl HasEllipticCurveOperations for BLS12381Curve {
    type BaseField = TF3;
    type UIntOrders = U384;

    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(4)
    }

    fn generator_affine_x() -> FieldElement<Self::BaseField> {
        todo!();
        //FieldElement::new([
        //    FieldElement::new(U384::from_be_hex("3685416753713387016781088315183077757961620795782546409894578378688607592378376318836054947676345821548104185464507")),
        //    FieldElement::new(U384::from_be_hex("0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"))
        //])
    }

    fn generator_affine_y() -> FieldElement<Self::BaseField> {
        todo!();
        //FieldElement::new([
        //    FieldElement::new(U384::from_be_hex("1339506544944476473020471379941921221584933875938349620426543736416511423956333506472724655353366534992391756441569")),
        //    FieldElement::new(U384::from_be_hex("0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"))
        //])
    }

    fn embedding_degree() -> u32 {
        2
    }

    fn order_r() -> Self::UIntOrders {
        order_r()
    }

    fn order_p() -> Self::UIntOrders {
        order_p()
    }

    fn target_normalization_power() -> Vec<u64> {
        todo!()
    }
}

impl IsCyclicBilinearGroup for EllipticCurveElement<BLS12381Curve> {
    type PairingOutput = FieldElement<<BLS12381Curve as HasEllipticCurveOperations>::BaseField>;

    fn generator() -> Self {
        Self::new([
            BLS12381Curve::generator_affine_x(),
            BLS12381Curve::generator_affine_y(),
            FieldElement::one(),
        ])
    }

    fn neutral_element() -> Self {
        Self::new(BLS12381Curve::neutral_element())
    }

    /// Computes the addition of `self` and `other`.
    /// Taken from "Moonmath" (Algorithm 7, page 89)
    fn operate_with(&self, other: &Self) -> Self {
        Self::new(BLS12381Curve::add(&self.value, &other.value))
    }

    fn pairing(&self, _other: &Self) -> Self::PairingOutput {
        todo!()
    }
}
