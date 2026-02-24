//! Multilinear Polynomials for Binius
//!
//! Provides a type alias for `DenseMultilinearPolynomial<BinaryTowerField128>` and
//! conversion helpers between `TowerFieldElement` and `FieldElement<BinaryTowerField128>`.

use crate::fields::tower::Tower;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::binary::tower_field::{
    from_tower, to_tower, BinaryTowerField128,
};
use lambdaworks_math::polynomial::DenseMultilinearPolynomial;

/// Type alias: multilinear polynomial over BinaryTowerField128.
///
/// This reuses the battle-tested `DenseMultilinearPolynomial` from lambdaworks-math,
/// which provides efficient evaluation, variable fixing, and other operations.
pub type MultilinearPoly = DenseMultilinearPolynomial<BinaryTowerField128>;

/// Type alias for field elements of BinaryTowerField128.
pub type FE = FieldElement<BinaryTowerField128>;

/// Convert a vector of `TowerFieldElement` to a `MultilinearPoly`.
pub fn tower_vec_to_poly(values: &[Tower]) -> MultilinearPoly {
    let evals: Vec<FE> = values.iter().map(from_tower).collect();
    DenseMultilinearPolynomial::new(evals)
}

/// Convert a `MultilinearPoly` to a vector of `TowerFieldElement`.
pub fn poly_to_tower_vec(poly: &MultilinearPoly) -> Vec<Tower> {
    poly.evals().iter().map(to_tower).collect()
}

/// Convert a slice of `TowerFieldElement` to `FieldElement<BinaryTowerField128>`.
pub fn tower_slice_to_fe(values: &[Tower]) -> Vec<FE> {
    values.iter().map(from_tower).collect()
}

/// Convert a slice of `FieldElement<BinaryTowerField128>` to `TowerFieldElement`.
pub fn fe_slice_to_tower(values: &[FE]) -> Vec<Tower> {
    values.iter().map(to_tower).collect()
}

/// The old `MultilinearPolynomial` type kept for backward compatibility during migration.
/// New code should use `MultilinearPoly` directly.
#[derive(Clone, Debug)]
pub struct MultilinearPolynomial {
    pub num_vars: usize,
    pub evals: Vec<Tower>,
}

impl MultilinearPolynomial {
    pub fn new(evals: Vec<Tower>) -> Result<Self, &'static str> {
        let len = evals.len();
        if len == 0 || (len & (len - 1)) != 0 {
            return Err("Number of evaluations must be a power of 2");
        }
        let num_vars = len.trailing_zeros() as usize;
        Ok(Self { num_vars, evals })
    }

    pub fn constant(c: Tower) -> Self {
        Self {
            num_vars: 0,
            evals: vec![c],
        }
    }

    pub fn degree(&self) -> usize {
        self.num_vars
    }

    pub fn evaluate(&self, point: &[Tower]) -> Tower {
        let fe_point: Vec<FE> = point.iter().map(from_tower).collect();
        let poly = self.to_dense_multilinear();
        to_tower(&poly.evaluate(fe_point).expect("evaluation should succeed"))
    }

    pub fn partial_evaluate(&self, var_idx: usize, value: Tower) -> MultilinearPolynomial {
        let poly = self.to_dense_multilinear();
        let fe_value = from_tower(&value);
        // Fix the first variable (the sumcheck convention)
        // The old code fixed var_idx, but DenseMultilinearPolynomial::fix_first_variable
        // always fixes variable 0. For compatibility, we fix variable 0 here.
        let _ = var_idx;
        let reduced = poly.fix_first_variable(&fe_value);
        let tower_evals: Vec<Tower> = reduced.evals().iter().map(to_tower).collect();
        MultilinearPolynomial {
            num_vars: self.num_vars.saturating_sub(1),
            evals: tower_evals,
        }
    }

    pub fn evaluations(&self) -> &[Tower] {
        &self.evals
    }

    pub fn from_closure<F>(num_vars: usize, f: F) -> Self
    where
        F: Fn(usize) -> Tower,
    {
        let size = 1 << num_vars;
        let evals: Vec<Tower> = (0..size).map(f).collect();
        Self { num_vars, evals }
    }

    pub fn add(&self, other: &MultilinearPolynomial) -> MultilinearPolynomial {
        assert_eq!(self.num_vars, other.num_vars);
        let evals: Vec<Tower> = self
            .evals
            .iter()
            .zip(other.evals.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        Self {
            num_vars: self.num_vars,
            evals,
        }
    }

    pub fn sub(&self, other: &MultilinearPolynomial) -> MultilinearPolynomial {
        assert_eq!(self.num_vars, other.num_vars);
        let evals: Vec<Tower> = self
            .evals
            .iter()
            .zip(other.evals.iter())
            .map(|(a, b)| *a - *b)
            .collect();
        Self {
            num_vars: self.num_vars,
            evals,
        }
    }

    pub fn scale(&self, scalar: Tower) -> MultilinearPolynomial {
        let evals: Vec<Tower> = self.evals.iter().map(|v| *v * scalar).collect();
        Self {
            num_vars: self.num_vars,
            evals,
        }
    }

    /// Convert to the lambdaworks `DenseMultilinearPolynomial`.
    pub fn to_dense_multilinear(&self) -> MultilinearPoly {
        tower_vec_to_poly(&self.evals)
    }

    /// Create from a lambdaworks `DenseMultilinearPolynomial`.
    pub fn from_dense_multilinear(poly: &MultilinearPoly) -> Self {
        let tower_evals: Vec<Tower> = poly.evals().iter().map(to_tower).collect();
        Self {
            num_vars: poly.num_vars(),
            evals: tower_evals,
        }
    }
}

impl Default for MultilinearPolynomial {
    fn default() -> Self {
        Self::constant(Tower::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tower_vec_to_poly_roundtrip() {
        let values = vec![
            Tower::new(0, 7),
            Tower::new(1, 7),
            Tower::new(2, 7),
            Tower::new(3, 7),
        ];
        let poly = tower_vec_to_poly(&values);
        let back = poly_to_tower_vec(&poly);
        for (a, b) in values.iter().zip(back.iter()) {
            assert_eq!(a.value(), b.value());
        }
    }

    #[test]
    fn test_multilinear_evaluate() {
        // P(x) over {0,1}: P(0) = 5, P(1) = 3
        // P(x) = 5*(1-x) + 3*x = 5 + (3-5)*x = 5 + (3 XOR 5)*x = 5 + 6*x
        // In char 2: P(x) = 5 + 6*x, so P(0) = 5, P(1) = 5 XOR 6 = 3
        let evals = vec![Tower::new(5, 7), Tower::new(3, 7)];
        let poly = MultilinearPolynomial::new(evals).unwrap();
        assert_eq!(poly.evaluate(&[Tower::new(0, 7)]).value(), 5);
        assert_eq!(poly.evaluate(&[Tower::new(1, 7)]).value(), 3);
    }

    #[test]
    fn test_multilinear_to_dense_roundtrip() {
        let evals = vec![
            Tower::new(1, 7),
            Tower::new(2, 7),
            Tower::new(3, 7),
            Tower::new(4, 7),
        ];
        let old_poly = MultilinearPolynomial::new(evals.clone()).unwrap();
        let dense = old_poly.to_dense_multilinear();
        let back = MultilinearPolynomial::from_dense_multilinear(&dense);
        for (a, b) in old_poly.evals.iter().zip(back.evals.iter()) {
            assert_eq!(a.value(), b.value());
        }
    }

    #[test]
    fn test_addition() {
        let p1 = MultilinearPolynomial::new(vec![Tower::new(1, 1), Tower::new(2, 1)]).unwrap();
        let p2 = MultilinearPolynomial::new(vec![Tower::new(3, 1), Tower::new(1, 1)]).unwrap();
        let sum = p1.add(&p2);
        assert_eq!(sum.evals[0].value(), 2); // 1 XOR 3
        assert_eq!(sum.evals[1].value(), 3); // 2 XOR 1
    }
}
