use crate::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    },
    unsigned_integer::element::U64,
};

pub type PallasMontgomeryBackendPrimeField<T> = MontgomeryBackendPrimeField<T, 4>;