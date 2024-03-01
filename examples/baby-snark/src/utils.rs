use std::ops::Neg;

use crate::common::FrElement;

pub fn i64_to_field(element: &i64) -> FrElement {
    let mut fr_element = FrElement::from(element.unsigned_abs());
    if element.is_negative() {
        fr_element = fr_element.neg()
    }

    fr_element
}

pub fn i64_vec_to_field(elements: &[i64]) -> Vec<FrElement> {
    elements.iter().map(i64_to_field).collect()
}

pub fn i64_matrix_to_field(elements: &[&[i64]]) -> Vec<Vec<FrElement>> {
    let mut matrix = Vec::new();
    for f in elements {
        matrix.push(i64_vec_to_field(f));
    }
    matrix
}
