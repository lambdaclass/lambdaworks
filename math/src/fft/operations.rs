use crate::{
    field::{
        element::FieldElement,
        traits::{IsField, IsTwoAdicField},
    },
    polynomial::Polynomial,
};

use super::{
    errors::FFTError,
    fft_cooley_tukey::{fft, inverse_fft},
};

pub fn evaluate_poly<F: IsField + IsTwoAdicField>(
    polynomial: Polynomial<FieldElement<F>>,
) -> Result<Vec<FieldElement<F>>, FFTError> {
    fft(polynomial.coefficients())
}

pub fn interpolate_poly<F: IsField + IsTwoAdicField>(
    polynomial: Polynomial<FieldElement<F>>,
) -> Result<Vec<FieldElement<F>>, FFTError> {
    inverse_fft(polynomial.coefficients())
}

pub fn evaluate_poly_with_offset<F: IsField + IsTwoAdicField>(
    polynomial: Polynomial<FieldElement<F>>,
    offset: &FieldElement<F>,
) -> Result<Vec<FieldElement<F>>, FFTError> {
    let scaled_polynomial = polynomial.scale(offset);
    fft(scaled_polynomial.coefficients())
}
