use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsFFTField;
use lambdaworks_math::field::traits::IsField;
use std::vec::Vec;

use super::domain::{CyclicDomain, CyclicDomainError};

#[derive(Debug, Clone)]
pub struct UnivariateLagrange<F: IsFFTField> {
    pub values: Vec<FieldElement<F>>,
    pub domain: CyclicDomain<F>,
}

impl<F: IsFFTField> UnivariateLagrange<F> {
    pub fn new(
        values: Vec<FieldElement<F>>,
        domain: CyclicDomain<F>,
    ) -> Result<Self, CyclicDomainError> {
        if values.len() != domain.size() {
            return Err(CyclicDomainError::SizeMismatch {
                expected: domain.size(),
                got: values.len(),
            });
        }
        Ok(Self { values, domain })
    }

    pub fn from_multilinear(values: Vec<FieldElement<F>>) -> Result<Self, CyclicDomainError> {
        let n = values.len();
        let log_n = n.ilog2() as usize;
        let domain = CyclicDomain::new(log_n)?;

        Ok(Self { values, domain })
    }

    pub fn n_variables(&self) -> usize {
        self.domain.n_variables
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

fn bit_reverse_indices<F: IsField>(values: &mut [FieldElement<F>]) {
    let n = values.len();
    let log_n = n.ilog2() as usize;

    for i in 1..n {
        let bits = usize::BITS as usize;
        let j = i.reverse_bits() >> (bits - log_n);
        if i < j {
            values.swap(i, j);
        }
    }
}

pub fn multilinear_to_univariate_fft<F: IsFFTField>(
    multilinear_evals: &[FieldElement<F>],
) -> Result<Vec<FieldElement<F>>, CyclicDomainError> {
    use lambdaworks_math::fft::cpu::ops::fft;
    use lambdaworks_math::fft::cpu::roots_of_unity::get_powers_of_primitive_root;
    use lambdaworks_math::field::traits::RootsConfig;

    let n = multilinear_evals.len();
    let log_n = n.ilog2() as usize;
    let _domain: CyclicDomain<F> = CyclicDomain::new(log_n)?;

    let mut values = multilinear_evals.to_vec();

    bit_reverse_indices(&mut values);

    let twiddles = get_powers_of_primitive_root::<F>(log_n as u64, n / 2, RootsConfig::BitReverse)
        .map_err(|_| CyclicDomainError::NoRootOfUnity(log_n))?;

    let fft_result =
        fft(&values, &twiddles).map_err(|_| CyclicDomainError::NoRootOfUnity(log_n))?;

    Ok(fft_result)
}

pub fn univariate_to_multilinear_fft<F: IsFFTField>(
    univariate_evals: &[FieldElement<F>],
) -> Result<Vec<FieldElement<F>>, CyclicDomainError> {
    use lambdaworks_math::fft::cpu::ops::fft;
    use lambdaworks_math::fft::cpu::roots_of_unity::get_powers_of_primitive_root;
    use lambdaworks_math::field::traits::RootsConfig;

    let n = univariate_evals.len();
    let log_n = n.ilog2() as usize;
    let _domain: CyclicDomain<F> = CyclicDomain::new(log_n)?;

    let mut values = univariate_evals.to_vec();

    bit_reverse_indices(&mut values);

    let inv_twiddles =
        get_powers_of_primitive_root::<F>(log_n as u64, n / 2, RootsConfig::BitReverseInversed)
            .map_err(|_| CyclicDomainError::NoRootOfUnity(log_n))?;

    let mut fft_result =
        fft(&values, &inv_twiddles).map_err(|_| CyclicDomainError::NoRootOfUnity(log_n))?;

    let n_inv = FieldElement::from(n as u64)
        .inv()
        .map_err(|_| CyclicDomainError::DomainSizeNotInvertible(n))?;
    for val in &mut fft_result {
        *val = val.clone() * n_inv.clone();
    }

    bit_reverse_indices(&mut fft_result);

    Ok(fft_result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::fft_friendly::babybear::Babybear31PrimeField;

    type F = Babybear31PrimeField;
    type FE = FieldElement<F>;

    #[test]
    fn test_univariate_lagrange_basic() {
        let values: Vec<FE> = (0..8).map(|i| FE::from(i as u64)).collect();
        let univariate = UnivariateLagrange::from_multilinear(values).unwrap();

        assert_eq!(univariate.n_variables(), 3);
        assert_eq!(univariate.len(), 8);
    }
}
