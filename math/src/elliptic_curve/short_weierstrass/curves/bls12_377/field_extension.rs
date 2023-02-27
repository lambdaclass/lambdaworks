use crate::field::{
    element::FieldElement,
    fields::u384_prime_field::{IsMontgomeryConfiguration, MontgomeryBackendPrimeField},
};
use crate::unsigned_integer::element::U384;

pub const BLS12377_PRIME_FIELD_ORDER: U384 = U384::from("1ae3a4617c510eac63b05c06ca1493b1a22d9f300f5138f1ef3622fba094800170b5d44300000008508c00000000001");

// FPBLS12377
#[derive(Clone, Debug)]
pub struct BLS12377FieldConfig;
impl IsMontgomeryConfiguration for BLS12377FieldConfig {
    const MODULUS: U384 = BLS12377_PRIME_FIELD_ORDER;
    const R2: U384 = U384::from("6dfccb1e914b88837e92f041790bf9bfdf7d03827dc3ac22a5f11162d6b46d0329fcaab00431b1b786686c9400cd22");
}

pub type BLS12377PrimeField = MontgomeryBackendPrimeField<BLS12377FieldConfig>;

impl FieldElement<BLS12377PrimeField> {
    pub fn new_base(a_hex: &str) -> Self {
        Self::new(U384::from(a_hex))
    }
}
