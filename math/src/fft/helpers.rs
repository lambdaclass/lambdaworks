use crate::field::{element::FieldElement, traits::IsField};

use super::errors::FFTError;

pub fn log2(n: usize) -> Result<u64, FFTError> {
    if !n.is_power_of_two() {
        return Err(FFTError::InvalidOrder(
            "The order of polynomial + 1 should a be power of 2".to_string(),
        ));
    }
    Ok(n.trailing_zeros() as u64)
}

pub fn pad_to_next_power_of_two<F: IsField>(a: &mut Vec<FieldElement<F>>) {
    let len = a.len() as f64;
    let b = len.log2() as u64;

    if 2_u64.pow(b as u32) == len as u64 {
        a.resize(len as usize, FieldElement::zero())
    }

    a.resize(2_u64.pow((b + 1) as u32) as usize, FieldElement::zero());
}
