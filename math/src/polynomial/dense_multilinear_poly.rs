use crate::field::element::FieldElement;
use crate::field::traits::{IsField, IsPrimeField};

/// Represents a multilinear polynomials as a collection of evaluations in lagrange basis

// TODO: add checks to track the max degree and number of variables.
#[derive(Debug, PartialEq, Clone)]
pub struct DenseMultilinearPolynomial<F: IsPrimeField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub evals: Vec<FieldElement<F>>,
    pub n_vars: usize, // number of variables
    pub len: usize,
}

pub trait Math {
    fn square_root(self) -> usize;
    fn pow2(self) -> usize;
    fn get_bits(self, num_bits: usize) -> Vec<bool>;
    fn log_2(self) -> usize;
}

impl Math for usize {
    #[inline]
    fn square_root(self) -> usize {
        (self as f64).sqrt() as usize
    }

    #[inline]
    fn pow2(self) -> usize {
        let base: usize = 2;
        base.pow(self as u32)
    }

    /// Returns the num_bits from n in a canonical order
    fn get_bits(self, num_bits: usize) -> Vec<bool> {
        (0..num_bits)
            .map(|shift_amount| ((self & (1 << (num_bits - shift_amount - 1))) > 0))
            .collect::<Vec<bool>>()
    }

    fn log_2(self) -> usize {
        assert_ne!(self, 0);

        if self.is_power_of_two() {
            (1usize.leading_zeros() - self.leading_zeros()) as usize
        } else {
            (0usize.leading_zeros() - self.leading_zeros()) as usize
        }
    }
}

impl<F: IsPrimeField> DenseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Build a new multilinear polynomial, from collection of multilinear monomials
    #[allow(dead_code)]
    pub fn new(evals: Vec<FieldElement<F>>) -> Self {
        // Pad non-power-2 evaluations to fill out the dense multilinear polynomial
        let mut poly_evals = evals;
        while !(poly_evals.len().is_power_of_two()) {
            poly_evals.push(FieldElement::zero());
        }

        DenseMultilinearPolynomial {
            evals: poly_evals.clone(),
            n_vars: poly_evals.len().log_2(),
            len: poly_evals.len(),
        }
    }

    pub fn num_vars(&self) -> usize {
        self.n_vars
    }

    pub fn evals(&self) -> &Vec<FieldElement<F>> {
        &self.evals
    }

    /// Evaluates `self` at the point `p` in O(n) time.
    /// Note: assumes p contains points for all variables aka is not sparse.
    // Ported from a16z/Lasso
    #[allow(dead_code)]
    pub fn evaluate(&self, r: Vec<FieldElement<F>>) -> FieldElement<F> {
        // r must have a value for each variable
        assert_eq!(r.len(), self.num_vars());

        let mut chis: Vec<FieldElement<F>> = vec![FieldElement::one(); r.len().pow2()];
        let mut size = 1;
        for j in 0..r.len() {
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                //TODO: eliminate clone
                let scalar = chis[i / 2].clone();
                chis[i] = &scalar * &r[j];
                chis[i - 1] = scalar - &chis[i];
            }
        }
        assert_eq!(chis.len(), self.evals.len());
        (0..self.evals.len())
            .map(|i| &self.evals[i] * &chis[i])
            .sum()
    }

    pub fn extend(&mut self, other: &DenseMultilinearPolynomial<F>) {
        assert_eq!(self.evals.len(), self.len);
        //TODO: eliminate clone
        let other_vec = other.evals.clone();
        assert_eq!(other_vec.len(), self.len);
        self.evals.extend(other_vec);
        self.n_vars += 1;
        self.len *= 2;
        assert_eq!(self.evals.len(), self.len);
    }

    pub fn merge(polys: &[DenseMultilinearPolynomial<F>]) -> DenseMultilinearPolynomial<F> {
        let mut Z: Vec<FieldElement<F>> = Vec::new();
        for poly in polys.iter() {
            Z.extend(poly.evals().clone().into_iter());
        }

        // pad the polynomial with zero polynomial at the end
        Z.resize(Z.len().next_power_of_two(), FieldElement::zero());

        DenseMultilinearPolynomial::new(Z)
    }

    pub fn from_usize(evals: &[usize]) -> Self {
        DenseMultilinearPolynomial::new(
            (0..evals.len())
                .map(|i| FieldElement::from(evals[i] as u64))
                .collect::<Vec<FieldElement<F>>>(),
        )
    }
}

#[cfg(test)]
mod tests {
    use rand::{random, rngs::StdRng, SeedableRng};
    use rand_chacha::rand_core::OsRng;

    use crate::field::fields::u64_prime_field::U64PrimeField;

    use super::*;

    const ORDER: u64 = 101;
    type F = U64PrimeField<ORDER>;
    type FE = FieldElement<F>;

    pub fn evals(r: Vec<FE>) -> Vec<FE> {
        let ell = r.len();

        let mut evals: Vec<FE> = vec![FE::one(); ell.pow2()];
        let mut size = 1;
        for j in 0..ell {
            // in each iteration, we double the size of chis
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                // copy each element from the prior iteration twice
                let scalar = evals[i / 2];
                evals[i] = scalar * r[j];
                evals[i - 1] = scalar - evals[i];
            }
        }
        evals
    }

    pub fn compute_factored_lens(ell: usize) -> (usize, usize) {
        (ell / 2, ell - ell / 2)
    }

    pub fn compute_factored_evals(r: Vec<FE>) -> (Vec<FE>, Vec<FE>) {
        let ell = r.len();
        let (left_num_vars, _right_num_vars) = compute_factored_lens(ell);

        let L = evals(r[..left_num_vars].to_vec());
        let R = evals(r[left_num_vars..ell].to_vec());

        (L, R)
    }

    fn evaluate_with_lr(Z: &[FE], r: &[FE]) -> FE {
        let (L, R) = compute_factored_evals(r.to_vec());

        let ell = r.len();
        // ensure ell is even
        assert!(ell % 2 == 0);
        // compute n = 2^\ell
        let n = ell.pow2();
        // compute m = sqrt(n) = 2^{\ell/2}
        let m = n.square_root();

        // compute vector-matrix product between L and Z viewed as a matrix
        let LZ = (0..m)
            .map(|i| {
                (0..m).fold(FE::zero(), |mut acc, j| {
                    acc += L[j] * Z[j * m + i];
                    acc
                })
            })
            .collect::<Vec<FE>>();

        // compute dot product between LZ and R
        (0..LZ.len()).map(|i| &LZ[i] * &R[i]).sum()
    }

    #[test]
    fn check_polynomial_evaluation() {
        // Z = [1, 2, 1, 4]
        let Z = vec![FE::one(), FE::from(2u64), FE::one(), FE::from(4u64)];

        // r = [4,3]
        let r = vec![FE::from(4u64), FE::from(3u64)];

        let eval_with_lr = evaluate_with_lr(&Z, &r);
        let poly = DenseMultilinearPolynomial::new(Z);

        let eval = poly.evaluate(r);
        assert_eq!(eval, FE::from(28u64));
        assert_eq!(eval_with_lr, eval);
    }

    /*
    #[test]
    fn test_add() {
        // polynomial 3t_1t_2 + 4t_2t_3
        let mut poly1 = SparseMultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((FE::new(3), vec![1, 2])),
            MultiLinearMonomial::new((FE::new(4), vec![2, 3])),
        ]);

        // polynomial 2t_1t_2 - 4t_2t_3
        let poly2 = SparseMultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((FE::new(2), vec![1, 2])),
            MultiLinearMonomial::new((-FE::new(4), vec![2, 3])),
        ]);

        // polynomial 5t_1t_2 + 6t_2t_3
        let expected = SparseMultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((FE::new(5), vec![1, 2])),
            MultiLinearMonomial::new((FE::new(0), vec![2, 3])),
        ]);

        poly1.add(poly2);
        assert_eq!(poly1, expected);

        // polynomial 2t_1t_2 - 4t_2t_3
        let poly2 = SparseMultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((FE::new(2), vec![1, 2])),
            MultiLinearMonomial::new((-FE::new(4), vec![2, 3])),
        ]);
        let mut poly_empty = SparseMultilinearPolynomial::<F>::new(vec![]);
        poly_empty.add(poly2);
        // polynomial 2t_1t_2 - 4t_2t_3
        let poly2 = SparseMultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((FE::new(2), vec![1, 2])),
            MultiLinearMonomial::new((-FE::new(4), vec![2, 3])),
        ]);
        assert_eq!(poly_empty, poly2);
    }

    #[test]
    fn test_add_monomial() {
        // polynomial 3t_1t_2 + 4t_2t_3
        let mut poly = SparseMultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((FE::new(3), vec![1, 2])),
            MultiLinearMonomial::new((FE::new(4), vec![2, 3])),
        ]);

        // monomial 3t_1t_2
        let mono = MultiLinearMonomial::new((FE::new(3), vec![1, 2]));

        // expected result 6t_1t_2 + 4t_2t_3
        let expected = SparseMultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((FE::new(6), vec![1, 2])),
            MultiLinearMonomial::new((FE::new(4), vec![2, 3])),
        ]);

        poly.add_monomial(&mono);
        assert_eq!(poly, expected);
    }

    #[test]
    fn test_partial_evaluation() {
        // 3ab + 4bc
        // partially evaluate b = 2
        // expected result = 6a + 8c
        // a = 0, b = 1, c = 2
        let poly = SparseMultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((FE::new(3), vec![0, 1])),
            MultiLinearMonomial::new((FE::new(4), vec![1, 2])),
        ]);
        assert_eq!(poly.n_vars, 3);
        let result = poly.partial_evaluate(&[(1, FE::new(2))]);
        assert_eq!(
            result,
            SparseMultilinearPolynomial {
                terms: vec![
                    MultiLinearMonomial {
                        coeff: FE::new(6),
                        vars: vec![0]
                    },
                    MultiLinearMonomial {
                        coeff: FE::new(8),
                        vars: vec![2]
                    }
                ],
                n_vars: 3,
            }
        );
    }

    #[test]
    fn test_all_vars_evaluation() {
        // 3abc + 4abc
        // evaluate: a = 1, b = 2, c = 3
        // expected result = 42
        // a = 0, b = 1, c = 2
        let poly = SparseMultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((FE::new(3), vec![0, 1, 2])),
            MultiLinearMonomial::new((FE::new(4), vec![0, 1, 2])),
        ]);
        let result = poly.evaluate(&[FE::one(), FE::new(2), FE::new(3)]);
        assert_eq!(result, FE::new(42));
    }

    #[test]
    fn test_partial_vars_evaluation() {
        // 3ab + 4bc
        // evaluate: a = 1, b = 2, c = 3
        // expected result = 30
        // a = 0, b = 1, c = 2
        let poly = SparseMultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((FE::new(3), vec![0, 1])),
            MultiLinearMonomial::new((FE::new(4), vec![1, 2])),
        ]);
        let result = poly.evaluate(&[FE::one(), FE::new(2), FE::new(3)]);
        assert_eq!(result, FE::new(30));
    }
    */
}
