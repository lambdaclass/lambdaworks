use cudarc::driver::DriverError;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{IsTwoAdicField, RootsConfig},
    },
    polynomial::Polynomial,
};

use super::ops::{fft, get_twiddles, log2};

pub fn evaluate_fft_cuda<F>(
    poly: &Polynomial<FieldElement<F>>,
) -> Result<Vec<FieldElement<F>>, DriverError>
where
    F: IsTwoAdicField,
    F::BaseType: Unpin,
{
    let order = log2(poly.coefficients.len()).unwrap();
    let twiddles = get_twiddles(order, RootsConfig::BitReverse).unwrap();

    fft(poly.coefficients(), &twiddles)
}

#[cfg(feature = "cuda")]
#[cfg(test)]
mod gpu_tests {}
