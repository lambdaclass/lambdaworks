use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsFFTField;
use std::vec::Vec;

#[derive(Debug, Clone)]
pub struct CyclicDomain<F: IsFFTField> {
    pub n_variables: usize,
    pub root: FieldElement<F>,
    pub inv_root: FieldElement<F>,
}

impl<F: IsFFTField> CyclicDomain<F> {
    pub fn new(n_variables: usize) -> Result<Self, CyclicDomainError> {
        if n_variables > F::TWO_ADICITY as usize {
            return Err(CyclicDomainError::InvalidOrder(n_variables));
        }

        let root = F::get_primitive_root_of_unity(n_variables as u64)
            .map_err(|_| CyclicDomainError::NoRootOfUnity(n_variables))?;

        let inv_root = root
            .inv()
            .map_err(|_| CyclicDomainError::NoRootOfUnity(n_variables))?;

        Ok(Self {
            n_variables,
            root,
            inv_root,
        })
    }

    pub fn size(&self) -> usize {
        1 << self.n_variables
    }

    pub fn get_point(&self, index: usize) -> FieldElement<F> {
        self.root.pow(index)
    }

    pub fn get_all_points(&self) -> Vec<FieldElement<F>> {
        (0..self.size()).map(|i| self.get_point(i)).collect()
    }
}

#[derive(Debug, Clone)]
pub enum CyclicDomainError {
    InvalidOrder(usize),
    NoRootOfUnity(usize),
    /// Domain size N is not invertible in the field (N >= char).
    DomainSizeNotInvertible(usize),
    /// Values length doesn't match domain size.
    SizeMismatch {
        expected: usize,
        got: usize,
    },
}

impl core::fmt::Display for CyclicDomainError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CyclicDomainError::InvalidOrder(n) => {
                write!(f, "Invalid order: {} exceeds field's 2-adicity", n)
            }
            CyclicDomainError::NoRootOfUnity(n) => {
                write!(f, "No primitive {}-th root of unity", 1 << n)
            }
            CyclicDomainError::DomainSizeNotInvertible(n) => {
                write!(f, "Domain size {n} is not invertible in the field")
            }
            CyclicDomainError::SizeMismatch { expected, got } => {
                write!(f, "values length mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl core::error::Error for CyclicDomainError {}

pub fn bits_of_index(index: usize, num_bits: usize) -> Vec<u64> {
    (0..num_bits).map(|i| ((index >> i) & 1) as u64).collect()
}

pub fn index_from_bits(bits: &[u64]) -> usize {
    bits.iter()
        .enumerate()
        .fold(0usize, |acc, (i, &bit)| acc | ((bit as usize) << i))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bits_of_index() {
        assert_eq!(bits_of_index(0, 3), vec![0, 0, 0]);
        assert_eq!(bits_of_index(1, 3), vec![1, 0, 0]);
        assert_eq!(bits_of_index(2, 3), vec![0, 1, 0]);
        assert_eq!(bits_of_index(3, 3), vec![1, 1, 0]);
        assert_eq!(bits_of_index(4, 3), vec![0, 0, 1]);
        assert_eq!(bits_of_index(7, 3), vec![1, 1, 1]);
    }

    #[test]
    fn test_index_from_bits() {
        assert_eq!(index_from_bits(&[0, 0, 0]), 0);
        assert_eq!(index_from_bits(&[1, 0, 0]), 1);
        assert_eq!(index_from_bits(&[0, 1, 0]), 2);
        assert_eq!(index_from_bits(&[1, 1, 0]), 3);
        assert_eq!(index_from_bits(&[0, 0, 1]), 4);
        assert_eq!(index_from_bits(&[1, 1, 1]), 7);
    }

    #[test]
    fn test_roundtrip_bits_index() {
        for i in 0..8 {
            let bits = bits_of_index(i, 3);
            let recovered = index_from_bits(&bits);
            assert_eq!(i, recovered);
        }
    }
}
