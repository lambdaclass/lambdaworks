use crate::{
    field::{element::FieldElement, traits::IsField},
    polynomial::{error::MultilinearError, Polynomial},
};
use alloc::{vec, vec::Vec};
use core::ops::{Add, Index, Mul};
#[cfg(feature = "parallel")]
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

/// Represents a multilinear polynomial as a vector of evaluations (FieldElements) in Lagrange basis.
///
/// # Mathematical Background
///
/// A multilinear polynomial in n variables is a polynomial where each variable appears with degree
/// at most 1. The unique multilinear extension (MLE) of a function f: {0,1}^n -> F is defined as:
///
/// ```text
/// f~(x_1, ..., x_n) = sum_{w in {0,1}^n} f(w) * prod_{i=1}^{n} (w_i * x_i + (1 - w_i) * (1 - x_i))
/// ```
///
/// This polynomial agrees with f on the Boolean hypercube {0,1}^n and is the unique multilinear
/// polynomial with this property.
///
/// # Representation
///
/// We store the evaluations on the Boolean hypercube in lexicographic order:
/// `[f(0,0,...,0), f(0,0,...,1), f(0,0,...,1,0), ..., f(1,1,...,1)]`
///
/// The index i corresponds to the binary representation of the point, where the least significant
/// bit corresponds to the last variable.
///
/// # References
///
/// - Thaler, "Proofs, Arguments, and Zero-Knowledge", Section 3.5
/// - GKR Protocol: Goldwasser, Kalai, Rothblum, "Delegating Computation"
/// - Spartan: Setty, "Spartan: Efficient and general-purpose zkSNARKs without trusted setup"
#[derive(Debug, PartialEq, Clone)]
pub struct DenseMultilinearPolynomial<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    evals: Vec<FieldElement<F>>,
    n_vars: usize,
    len: usize,
}

impl<F: IsField> DenseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Constructs a new multilinear polynomial from a collection of evaluations.
    /// Pads non-power-of-2 evaluations with zeros.
    pub fn new(mut evals: Vec<FieldElement<F>>) -> Self {
        while !evals.len().is_power_of_two() {
            evals.push(FieldElement::zero());
        }
        let len = evals.len();
        DenseMultilinearPolynomial {
            n_vars: log_2(len),
            evals,
            len,
        }
    }

    /// Returns the number of variables.
    pub fn num_vars(&self) -> usize {
        self.n_vars
    }

    /// Returns a reference to the evaluations vector.
    pub fn evals(&self) -> &Vec<FieldElement<F>> {
        &self.evals
    }

    /// Returns the total number of evaluations (2^num_vars).
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Evaluates `self` at the point `r` (a vector of FieldElements) in O(n) time.
    /// `r` must have a value for each variable.
    pub fn evaluate(&self, r: Vec<FieldElement<F>>) -> Result<FieldElement<F>, MultilinearError> {
        if r.len() != self.num_vars() {
            return Err(MultilinearError::IncorrectNumberofEvaluationPoints(
                r.len(),
                self.num_vars(),
            ));
        }
        let mut chis: Vec<FieldElement<F>> =
            vec![FieldElement::one(); (2usize).pow(r.len() as u32)];
        let mut size = 1;
        for j in r {
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                let half_i = i / 2;
                let temp = &chis[half_i] * &j;
                chis[i] = temp;
                chis[i - 1] = &chis[half_i] - &chis[i];
            }
        }
        #[cfg(feature = "parallel")]
        let iter = (0..chis.len()).into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = 0..chis.len();
        Ok(iter.map(|i| &self.evals[i] * &chis[i]).sum())
    }

    /// Evaluates a slice of evaluations with the given point `r`.
    pub fn evaluate_with(
        evals: &[FieldElement<F>],
        r: &[FieldElement<F>],
    ) -> Result<FieldElement<F>, MultilinearError> {
        let mut chis: Vec<FieldElement<F>> =
            vec![FieldElement::one(); (2usize).pow(r.len() as u32)];
        if chis.len() != evals.len() {
            return Err(MultilinearError::ChisAndEvalsLengthMismatch(
                chis.len(),
                evals.len(),
            ));
        }
        let mut size = 1;
        for j in r {
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                let half_i = i / 2;
                let temp = &chis[half_i] * j;
                chis[i] = temp;
                chis[i - 1] = &chis[half_i] - &chis[i];
            }
        }
        Ok((0..evals.len()).map(|i| &evals[i] * &chis[i]).sum())
    }

    /// Fixes the first variable to the given value `r` and returns a new DenseMultilinearPolynomial
    /// with one fewer variable.
    ///
    /// Combines each pair of evaluations as: new_eval = a + r * (b - a)
    ///  This reduces the polynomial by one variable, allowing it to later be collapsed
    /// into a univariate polynomial by summing over the remaining variables.
    ///
    /// Example (2 variables): evaluations are ordered as:
    ///     [f(0,0), f(0,1), f(1,0), f(1,1)]
    /// Fixing the first variable `x = r` produces evaluations of a 1-variable polynomial:
    ///     [f(r,0), f(r,1)]
    /// computed explicitly as:
    ///     f(r,0) = f(0,0) + r * ( f(1,0) - f(0,0)),
    ///     f(r,1) = f(0,1) + r * (f(1,1) - f(0,1))
    pub fn fix_first_variable(&self, r: &FieldElement<F>) -> DenseMultilinearPolynomial<F> {
        let n = self.num_vars();
        assert!(n > 0, "Cannot fix variable in a 0-variable polynomial");
        let half = 1 << (n - 1);
        let new_evals: Vec<FieldElement<F>> = (0..half)
            .map(|j| {
                let a = &self.evals[j];
                let b = &self.evals[j + half];
                a + r * (b - a)
            })
            .collect();
        DenseMultilinearPolynomial::from((n - 1, new_evals))
    }

    /// Partially evaluates the polynomial by fixing one variable at the specified index.
    ///
    /// Given f(x_0, x_1, ..., x_{n-1}) and fixing x_{var_index} = value, returns
    /// g(x_0, ..., x_{var_index-1}, x_{var_index+1}, ..., x_{n-1}) = f(..., value, ...)
    ///
    /// This is a key operation in the sumcheck protocol, where variables are fixed
    /// one at a time based on verifier challenges.
    ///
    /// # Mathematical Definition
    ///
    /// For a multilinear polynomial f and a value r in F:
    /// ```text
    /// partial_eval(f, i, r)(x_0, ..., x_{n-2}) = f(x_0, ..., x_{i-1}, r, x_i, ..., x_{n-2})
    /// ```
    ///
    /// # Arguments
    /// * `var_index` - The index of the variable to fix (0-indexed from left)
    /// * `value` - The field element value to assign to the variable
    ///
    /// # Panics
    /// Panics if `var_index >= self.num_vars()` or if the polynomial has 0 variables.
    ///
    /// # Example
    /// ```ignore
    /// // f(x, y, z) evaluated at various boolean points
    /// let f = DenseMultilinearPolynomial::new(vec![...]);
    /// // Fix y = r, resulting in g(x, z)
    /// let g = f.partial_evaluate(1, &r);
    /// ```
    pub fn partial_evaluate(
        &self,
        var_index: usize,
        value: &FieldElement<F>,
    ) -> DenseMultilinearPolynomial<F> {
        let n = self.num_vars();
        assert!(n > 0, "Cannot partially evaluate a 0-variable polynomial");
        assert!(
            var_index < n,
            "Variable index {} out of bounds for {}-variable polynomial",
            var_index,
            n
        );

        // Strategy: For each evaluation point in the new polynomial, we interpolate
        // between the two relevant points that differ only in the var_index position.

        let new_len = 1 << (n - 1);
        let mut new_evals = Vec::with_capacity(new_len);

        // The bit position in the original index corresponding to var_index
        // Variables are indexed left-to-right: x_0 is MSB, x_{n-1} is LSB
        let bit_pos = n - 1 - var_index;

        for new_idx in 0..new_len {
            // Compute the two original indices that correspond to this new index
            // with the var_index bit set to 0 and 1 respectively.

            // Insert a 0 bit at position bit_pos
            let low_mask = (1 << bit_pos) - 1;
            let high_part = (new_idx >> bit_pos) << (bit_pos + 1);
            let low_part = new_idx & low_mask;
            let idx_0 = high_part | low_part;
            let idx_1 = idx_0 | (1 << bit_pos);

            // Linear interpolation: f(0) + value * (f(1) - f(0))
            let a = &self.evals[idx_0];
            let b = &self.evals[idx_1];
            new_evals.push(a + value * (b - a));
        }

        DenseMultilinearPolynomial::from((n - 1, new_evals))
    }

    /// Partially evaluates the polynomial by fixing multiple variables at once.
    ///
    /// This is more efficient than repeated single partial evaluations when fixing
    /// multiple variables, as it avoids creating intermediate polynomials.
    ///
    /// # Arguments
    /// * `assignments` - Slice of (variable_index, value) pairs. Variable indices must be
    ///   unique and in ascending order for well-defined behavior.
    ///
    /// # Mathematical Definition
    ///
    /// Given assignments [(i_1, r_1), (i_2, r_2), ...], computes:
    /// ```text
    /// f(x_0, ..., x_{i_1-1}, r_1, x_{i_1}, ..., x_{i_2-1}, r_2, ...)
    /// ```
    ///
    /// # Panics
    /// Panics if any variable index is out of bounds.
    ///
    /// # Example
    /// ```ignore
    /// // f(x, y, z, w) -> g(y, w) by fixing x=r1, z=r2
    /// let g = f.partial_evaluate_many(&[(0, r1), (2, r2)]);
    /// ```
    pub fn partial_evaluate_many(
        &self,
        assignments: &[(usize, FieldElement<F>)],
    ) -> DenseMultilinearPolynomial<F> {
        if assignments.is_empty() {
            return self.clone();
        }

        let n = self.num_vars();
        for (var_idx, _) in assignments {
            assert!(
                *var_idx < n,
                "Variable index {} out of bounds for {}-variable polynomial",
                var_idx,
                n
            );
        }

        // Sort assignments by variable index (descending) to fix from right to left.
        // By processing higher indices first, we ensure that removing a variable
        // doesn't affect the indices of variables we haven't processed yet.
        //
        // Example: f(x0, x1, x2) with assignments [(0, a), (2, b)]
        // Process index 2 first: f(x0, x1, x2) -> g(x0, x1) = f(x0, x1, b)
        // Process index 0 next: g(x0, x1) -> h(x1) = g(a, x1)
        // Variable x0 is still at index 0 in g because we removed x2 (higher index)
        let mut sorted_assignments: Vec<_> = assignments.to_vec();
        sorted_assignments.sort_by(|a, b| b.0.cmp(&a.0));

        let mut current = self.clone();
        for (original_idx, value) in sorted_assignments {
            // Since we process in descending order, the variable at original_idx
            // remains at that same index in the current polynomial until we process it.
            // This is because removing a higher-indexed variable doesn't shift lower indices.
            current = current.partial_evaluate(original_idx, &value);
        }

        current
    }

    /// Evaluates the polynomial at multiple points efficiently.
    ///
    /// When evaluating at many points, this method can share computation across
    /// evaluations for better performance compared to calling `evaluate()` repeatedly.
    ///
    /// # Arguments
    /// * `points` - Slice of points, where each point is a vector of field elements
    ///   of length equal to `num_vars()`.
    ///
    /// # Returns
    /// A vector of evaluations, one for each input point.
    ///
    /// # Errors
    /// Returns an error if any point has incorrect length.
    ///
    /// # Example
    /// ```ignore
    /// let poly = DenseMultilinearPolynomial::new(evals);
    /// let points = vec![vec![r1, r2], vec![s1, s2]];
    /// let results = poly.batch_evaluate(&points)?;
    /// // results[0] = poly.evaluate(vec![r1, r2])
    /// // results[1] = poly.evaluate(vec![s1, s2])
    /// ```
    pub fn batch_evaluate(
        &self,
        points: &[Vec<FieldElement<F>>],
    ) -> Result<Vec<FieldElement<F>>, MultilinearError> {
        // Validate all points first
        for point in points {
            if point.len() != self.num_vars() {
                return Err(MultilinearError::IncorrectNumberofEvaluationPoints(
                    point.len(),
                    self.num_vars(),
                ));
            }
        }

        // For parallel execution, process evaluations concurrently
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            // All points have been validated to have correct length above,
            // so evaluate() will not return an error for dimension mismatch.
            let results: Result<Vec<_>, _> = points
                .par_iter()
                .map(|point| self.evaluate(point.clone()))
                .collect();
            results
        }

        #[cfg(not(feature = "parallel"))]
        {
            points
                .iter()
                .map(|point| self.evaluate(point.clone()))
                .collect()
        }
    }

    /// Computes the tensor product (Kronecker product) with another multilinear polynomial.
    ///
    /// Given f(x_1, ..., x_n) and g(y_1, ..., y_m), computes h(x_1, ..., x_n, y_1, ..., y_m)
    /// where h(x, y) = f(x) * g(y).
    ///
    /// # Mathematical Definition
    ///
    /// The tensor product h = f tensor g is defined such that for any (x, y) in {0,1}^{n+m}:
    /// ```text
    /// h(x_1, ..., x_n, y_1, ..., y_m) = f(x_1, ..., x_n) * g(y_1, ..., y_m)
    /// ```
    ///
    /// The evaluation vector of h is the Kronecker product of the evaluation vectors:
    /// ```text
    /// evals(h) = evals(f) tensor evals(g)
    /// ```
    ///
    /// # Arguments
    /// * `other` - The polynomial to take the tensor product with
    ///
    /// # Returns
    /// A new polynomial with `self.num_vars() + other.num_vars()` variables.
    ///
    /// # Example
    /// ```ignore
    /// let f = DenseMultilinearPolynomial::new(vec![a, b]); // 1 variable
    /// let g = DenseMultilinearPolynomial::new(vec![c, d]); // 1 variable
    /// let h = f.tensor_product(&g); // 2 variables
    /// // h.evals() = [a*c, a*d, b*c, b*d]
    /// ```
    pub fn tensor_product(&self, other: &Self) -> Self {
        let new_num_vars = self.n_vars + other.n_vars;
        let new_len = self.len * other.len;
        let mut new_evals = Vec::with_capacity(new_len);

        // Kronecker product: for each eval in self, multiply by all evals in other
        for self_eval in &self.evals {
            for other_eval in &other.evals {
                new_evals.push(self_eval * other_eval);
            }
        }

        DenseMultilinearPolynomial {
            evals: new_evals,
            n_vars: new_num_vars,
            len: new_len,
        }
    }

    /// Returns the evaluations of the polynomial on the Boolean hypercube \(\{0,1\}^n\).
    /// Since we are in Lagrange basis, this is just the elements stored in self.evals.
    pub fn to_evaluations(&self) -> Vec<FieldElement<F>> {
        self.evals.clone()
    }

    /// Collapses the last variable by fixing it to 0 and 1,
    /// sums the evaluations, and returns a univariate polynomial (as a Polynomial)
    /// of the form: sum0 + (sum1 - sum0) * x.
    pub fn to_univariate(&self) -> Polynomial<FieldElement<F>> {
        let poly0 = self.fix_first_variable(&FieldElement::zero());
        let poly1 = self.fix_first_variable(&FieldElement::one());
        let sum0: FieldElement<F> = poly0.to_evaluations().into_iter().sum();
        let sum1: FieldElement<F> = poly1.to_evaluations().into_iter().sum();
        let diff = sum1 - &sum0;
        Polynomial::new(&[sum0, diff])
    }

    /// Multiplies the polynomial by a scalar.
    pub fn scalar_mul(&self, scalar: &FieldElement<F>) -> Self {
        let mut new_poly = self.clone();
        new_poly.evals.iter_mut().for_each(|eval| *eval *= scalar);
        new_poly
    }

    /// Extends this DenseMultilinearPolynomial by concatenating another polynomial of the same length.
    pub fn extend(&mut self, other: &DenseMultilinearPolynomial<F>) {
        debug_assert_eq!(self.evals.len(), self.len);
        debug_assert_eq!(other.evals.len(), self.len);
        self.evals.extend(other.evals.iter().cloned());
        self.n_vars += 1;
        self.len *= 2;
        debug_assert_eq!(self.evals.len(), self.len);
    }

    /// Merges a series of DenseMultilinearPolynomials into one polynomial by concatenating
    /// their evaluation vectors in order.
    /// Zero-pads the final merged polynomial to the next power-of-two length if necessary.
    pub fn merge(polys: &[DenseMultilinearPolynomial<F>]) -> DenseMultilinearPolynomial<F> {
        // Calculate total size needed for pre-allocation
        let total_len: usize = polys.iter().map(|p| p.evals.len()).sum();
        let final_len = total_len.next_power_of_two();
        let mut z: Vec<FieldElement<F>> = Vec::with_capacity(final_len);
        for poly in polys {
            z.extend(poly.evals.iter().cloned());
        }
        z.resize(final_len, FieldElement::zero());
        DenseMultilinearPolynomial::new(z)
    }

    /// Constructs a DenseMultilinearPolynomial from a slice of u64 values.
    pub fn from_u64(evals: &[u64]) -> Self {
        DenseMultilinearPolynomial::new(evals.iter().map(|&i| FieldElement::from(i)).collect())
    }

    /// Constructs a DenseMultilinearPolynomial from evaluations on the Boolean hypercube.
    ///
    /// This is an alias for `new()` with additional documentation about the multilinear
    /// extension property.
    ///
    /// # Mathematical Background
    ///
    /// Given evaluations [f(0,0,...,0), f(0,0,...,1), ..., f(1,1,...,1)] on the Boolean
    /// hypercube {0,1}^n, this constructs the unique multilinear polynomial that agrees
    /// with these values. The polynomial is stored in evaluation form (Lagrange basis).
    ///
    /// # Arguments
    /// * `evals` - Vector of 2^n field elements representing evaluations. If not a power
    ///   of 2, will be padded with zeros.
    ///
    /// # Example
    /// ```ignore
    /// // f(0,0) = 1, f(0,1) = 2, f(1,0) = 3, f(1,1) = 4
    /// let f = DenseMultilinearPolynomial::from_evaluations(vec![
    ///     FE::from(1), FE::from(2), FE::from(3), FE::from(4)
    /// ]);
    /// // f(x, y) = 1 + x + y + xy
    /// assert_eq!(f.evaluate(vec![FE::zero(), FE::zero()]).unwrap(), FE::from(1));
    /// assert_eq!(f.evaluate(vec![FE::one(), FE::one()]).unwrap(), FE::from(4));
    /// ```
    pub fn from_evaluations(evals: Vec<FieldElement<F>>) -> Self {
        Self::new(evals)
    }
}

/// Constructs the equality polynomial eq(x, r) as a multilinear polynomial.
///
/// The equality polynomial is defined as:
/// ```text
/// eq(x, r) = prod_{i=1}^{n} (x_i * r_i + (1 - x_i) * (1 - r_i))
/// ```
///
/// This polynomial equals 1 when x = r on the Boolean hypercube {0,1}^n, and 0 when x != r.
/// It is a fundamental building block in multilinear sumcheck protocols.
///
/// # Mathematical Properties
///
/// - eq(w, r) = 1 if w = r (for w in {0,1}^n)
/// - eq(w, r) = 0 if w != r (for w in {0,1}^n)
/// - sum_{w in {0,1}^n} eq(w, r) = 1 for any r
///
/// # Arguments
/// * `point` - The point r = (r_1, ..., r_n) to fix in the equality polynomial
///
/// # Returns
/// A multilinear polynomial whose evaluations on {0,1}^n form a one-hot encoding
/// based on the binary representation of r.
///
/// # References
/// - Thaler, "Proofs, Arguments, and Zero-Knowledge", Section 4.1
/// - Used extensively in GKR protocol and Spartan
///
/// # Example
/// ```ignore
/// let r = vec![FE::from(2), FE::from(3)]; // Some random field elements
/// let eq_poly = eq_polynomial::<F>(&r);
/// // eq_poly(0, 0) evaluates to (1-2)*(1-3) = 2 (in the field)
/// // eq_poly(1, 1) evaluates to 2*3 = 6
/// ```
pub fn eq_polynomial<F: IsField>(point: &[FieldElement<F>]) -> DenseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    let num_vars = point.len();
    let len = 1 << num_vars;
    let mut evals = Vec::with_capacity(len);

    // For each point w in {0,1}^n, compute eq(w, point)
    for w in 0..len {
        let mut prod = FieldElement::<F>::one();
        for (i, r_i) in point.iter().enumerate() {
            // w_i is the i-th bit of w (MSB first, so x_0 corresponds to highest bit)
            let w_i = (w >> (num_vars - 1 - i)) & 1;

            // eq_i = w_i * r_i + (1 - w_i) * (1 - r_i)
            if w_i == 1 {
                prod *= r_i;
            } else {
                prod *= FieldElement::<F>::one() - r_i;
            }
        }
        evals.push(prod);
    }

    DenseMultilinearPolynomial::from((num_vars, evals))
}

/// Evaluates the equality polynomial eq(x, r) at a single point x.
///
/// This is more efficient than constructing the full eq polynomial when only
/// a single evaluation is needed.
///
/// # Mathematical Definition
///
/// ```text
/// eq(x, r) = prod_{i=1}^{n} (x_i * r_i + (1 - x_i) * (1 - r_i))
/// ```
///
/// # Arguments
/// * `x` - The point x at which to evaluate
/// * `r` - The fixed point r in the equality polynomial
///
/// # Panics
/// Panics if `x.len() != r.len()`.
///
/// # Example
/// ```ignore
/// let x = vec![FE::from(1), FE::from(0)];
/// let r = vec![FE::from(1), FE::from(0)];
/// assert_eq!(eq_eval(&x, &r), FE::one()); // x == r on boolean hypercube
/// ```
pub fn eq_eval<F: IsField>(x: &[FieldElement<F>], r: &[FieldElement<F>]) -> FieldElement<F> {
    assert_eq!(
        x.len(),
        r.len(),
        "Points must have the same dimension: x has {} vars, r has {} vars",
        x.len(),
        r.len()
    );

    let mut result = FieldElement::<F>::one();
    for (x_i, r_i) in x.iter().zip(r.iter()) {
        // eq_i = x_i * r_i + (1 - x_i) * (1 - r_i)
        //      = x_i * r_i + 1 - x_i - r_i + x_i * r_i
        //      = 2 * x_i * r_i - x_i - r_i + 1
        let term = x_i * r_i + (FieldElement::<F>::one() - x_i) * (FieldElement::<F>::one() - r_i);
        result *= term;
    }
    result
}

impl<F: IsField> Index<usize> for DenseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = FieldElement<F>;

    #[inline(always)]
    fn index(&self, index: usize) -> &FieldElement<F> {
        &self.evals[index]
    }
}

/// Adds two DenseMultilinearPolynomials.
/// Assumes that both polynomials have the same number of variables.
impl<F: IsField> Add for DenseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = Result<Self, &'static str>;

    fn add(self, other: Self) -> Self::Output {
        if self.num_vars() != other.num_vars() {
            return Err("Polynomials must have the same number of variables");
        }
        #[cfg(feature = "parallel")]
        let evals = self.evals.into_par_iter().zip(other.evals.into_par_iter());
        #[cfg(not(feature = "parallel"))]
        let evals = self.evals.iter().zip(other.evals.iter());
        let sum: Vec<FieldElement<F>> = evals.map(|(a, b)| a + b).collect();
        Ok(DenseMultilinearPolynomial::new(sum))
    }
}

impl<F: IsField> Mul<FieldElement<F>> for DenseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = DenseMultilinearPolynomial<F>;

    fn mul(self, rhs: FieldElement<F>) -> Self::Output {
        Self::scalar_mul(&self, &rhs)
    }
}

impl<F: IsField> Mul<&FieldElement<F>> for DenseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = DenseMultilinearPolynomial<F>;

    fn mul(self, rhs: &FieldElement<F>) -> Self::Output {
        Self::scalar_mul(&self, rhs)
    }
}

/// Helper function to calculate log₂(n).
fn log_2(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    if n.is_power_of_two() {
        (1usize.leading_zeros() - n.leading_zeros()) as usize
    } else {
        (0usize.leading_zeros() - n.leading_zeros()) as usize
    }
}

impl<F: IsField> From<(usize, Vec<FieldElement<F>>)> for DenseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    fn from((num_vars, evaluations): (usize, Vec<FieldElement<F>>)) -> Self {
        assert_eq!(
            evaluations.len(),
            1 << num_vars,
            "The size of evaluations should be 2^num_vars."
        );
        DenseMultilinearPolynomial {
            n_vars: num_vars,
            evals: evaluations,
            len: 1 << num_vars,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::fields::u64_prime_field::U64PrimeField;
    const ORDER: u64 = 101;
    type F = U64PrimeField<ORDER>;
    type FE = FieldElement<F>;

    pub fn evals(r: Vec<FE>) -> Vec<FE> {
        let mut evals: Vec<FE> = vec![FE::one(); (2usize).pow(r.len() as u32)];
        let mut size = 1;
        for j in r {
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                let scalar = evals[i / 2];
                evals[i] = scalar * j;
                evals[i - 1] = scalar - evals[i];
            }
        }
        evals
    }

    pub fn compute_factored_evals(r: Vec<FE>) -> (Vec<FE>, Vec<FE>) {
        let size = r.len();
        let (left_num_vars, _right_num_vars) = (size / 2, size - size / 2);
        let l = evals(r[..left_num_vars].to_vec());
        let r = evals(r[left_num_vars..size].to_vec());
        (l, r)
    }

    fn evaluate_with_lr(z: &[FE], r: &[FE]) -> FE {
        let (l, r) = compute_factored_evals(r.to_vec());
        let size = r.len();
        // Ensure size is even.
        assert!(size % 2 == 0);
        // n = 2^size
        let n = (2usize).pow(size as u32);
        // Compute m = sqrt(n) = 2^(l/2)
        let m = (n as f64).sqrt() as usize;
        // Compute vector-matrix product between L and Z (viewed as a matrix)
        let lz = (0..m)
            .map(|i| {
                (0..m).fold(FE::zero(), |mut acc, j| {
                    acc += l[j] * z[j * m + i];
                    acc
                })
            })
            .collect::<Vec<FE>>();
        // Compute dot product between LZ and R
        (0..lz.len()).map(|i| lz[i] * r[i]).sum()
    }

    #[test]
    fn evaluation() {
        // Example: Z = [1, 2, 1, 4]
        let z = vec![FE::one(), FE::from(2u64), FE::one(), FE::from(4u64)];
        // r = [4, 3]
        let r = vec![FE::from(4u64), FE::from(3u64)];
        let eval_with_lr = evaluate_with_lr(&z, &r);
        let poly = DenseMultilinearPolynomial::new(z);
        let eval = poly.evaluate(r).unwrap();
        assert_eq!(eval, FE::from(28u64));
        assert_eq!(eval_with_lr, eval);
    }

    #[test]
    fn evaluate_with() {
        let two = FE::from(2);
        let z = vec![
            FE::zero(),
            FE::zero(),
            FE::zero(),
            FE::one(),
            FE::one(),
            FE::one(),
            FE::zero(),
            two,
        ];
        let x = vec![FE::one(), FE::one(), FE::one()];
        let y = DenseMultilinearPolynomial::<F>::evaluate_with(z.as_slice(), x.as_slice()).unwrap();
        assert_eq!(y, two);
    }

    #[test]
    fn add() {
        let a = DenseMultilinearPolynomial::new(vec![FE::from(3); 4]);
        let b = DenseMultilinearPolynomial::new(vec![FE::from(7); 4]);
        let c = a.add(b).unwrap();
        assert_eq!(*c.evals(), vec![FE::from(10); 4]);
    }

    #[test]
    fn mul() {
        let a = DenseMultilinearPolynomial::new(vec![FE::from(3); 4]);
        let b = a.mul(&FE::from(2));
        assert_eq!(*b.evals(), vec![FE::from(6); 4]);
    }

    // Take a multilinear polynomial of length 2^2 and merge with a polynomial of 2^1.
    // The resulting polynomial should be padded to length 2^3 = 8 and the last two evaluations should be FE::zero().
    #[test]
    fn merge() {
        let a = DenseMultilinearPolynomial::new(vec![FE::from(3); 4]);
        let b = DenseMultilinearPolynomial::new(vec![FE::from(2); 2]);
        let c = DenseMultilinearPolynomial::merge(&[a, b]);
        assert_eq!(c.len(), 8);
        assert_eq!(c[c.len() - 1], FE::zero());
        assert_eq!(c[c.len() - 2], FE::zero());
        let d = DenseMultilinearPolynomial::new(vec![
            FE::from(3),
            FE::from(3),
            FE::from(3),
            FE::from(3),
            FE::from(2),
            FE::from(2),
            FE::zero(),
            FE::zero(),
        ]);
        assert_eq!(c, d);
    }

    #[test]
    fn extend() {
        let mut a = DenseMultilinearPolynomial::new(vec![FE::from(3); 4]);
        let b = DenseMultilinearPolynomial::new(vec![FE::from(3); 4]);
        a.extend(&b);
        assert_eq!(a.len(), 8);
        assert_eq!(a.num_vars(), 3);
    }

    #[test]
    #[should_panic]
    fn extend_unequal() {
        let mut a = DenseMultilinearPolynomial::new(vec![FE::from(3); 4]);
        let b = DenseMultilinearPolynomial::new(vec![FE::from(3); 2]);
        a.extend(&b);
    }

    // ============================================
    // Tests for new multilinear polynomial operations
    // ============================================

    #[test]
    fn partial_evaluate_first_variable() {
        // f(x, y) with evals [f(0,0), f(0,1), f(1,0), f(1,1)] = [1, 2, 3, 4]
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
        ]);

        // Fix x = 0, should give g(y) = [f(0,0), f(0,1)] = [1, 2]
        let g_at_0 = poly.partial_evaluate(0, &FE::zero());
        assert_eq!(g_at_0.num_vars(), 1);
        assert_eq!(*g_at_0.evals(), vec![FE::from(1), FE::from(2)]);

        // Fix x = 1, should give g(y) = [f(1,0), f(1,1)] = [3, 4]
        let g_at_1 = poly.partial_evaluate(0, &FE::one());
        assert_eq!(g_at_1.num_vars(), 1);
        assert_eq!(*g_at_1.evals(), vec![FE::from(3), FE::from(4)]);

        // Verify partial_evaluate(0, r) matches fix_first_variable(r)
        let r = FE::from(7);
        let via_partial = poly.partial_evaluate(0, &r);
        let via_fix_first = poly.fix_first_variable(&r);
        assert_eq!(via_partial, via_fix_first);
    }

    #[test]
    fn partial_evaluate_second_variable() {
        // f(x, y) with evals [f(0,0), f(0,1), f(1,0), f(1,1)] = [1, 2, 3, 4]
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
        ]);

        // Fix y = 0, should give g(x) = [f(0,0), f(1,0)] = [1, 3]
        let g_at_0 = poly.partial_evaluate(1, &FE::zero());
        assert_eq!(g_at_0.num_vars(), 1);
        assert_eq!(*g_at_0.evals(), vec![FE::from(1), FE::from(3)]);

        // Fix y = 1, should give g(x) = [f(0,1), f(1,1)] = [2, 4]
        let g_at_1 = poly.partial_evaluate(1, &FE::one());
        assert_eq!(g_at_1.num_vars(), 1);
        assert_eq!(*g_at_1.evals(), vec![FE::from(2), FE::from(4)]);
    }

    #[test]
    fn partial_evaluate_correctness() {
        // f(x, y, z) with 8 evaluations
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
            FE::from(5),
            FE::from(6),
            FE::from(7),
            FE::from(8),
        ]);

        let r = FE::from(5);

        // Fixing middle variable y = r
        let g = poly.partial_evaluate(1, &r);

        // Verify by evaluating original at specific points
        // g(x, z) should equal f(x, r, z)
        for x in [FE::zero(), FE::one()] {
            for z in [FE::zero(), FE::one()] {
                let g_eval = g
                    .evaluate(vec![x.clone(), z.clone()])
                    .expect("Valid evaluation point for g");
                let f_eval = poly
                    .evaluate(vec![x.clone(), r.clone(), z.clone()])
                    .expect("Valid evaluation point for f");
                assert_eq!(g_eval, f_eval);
            }
        }
    }

    #[test]
    fn partial_evaluate_many_basic() {
        // f(x, y, z) with 8 evaluations
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
            FE::from(5),
            FE::from(6),
            FE::from(7),
            FE::from(8),
        ]);

        let r1 = FE::from(3);
        let r2 = FE::from(7);

        // Fix x = r1, z = r2, leaving only y
        let g = poly.partial_evaluate_many(&[(0, r1.clone()), (2, r2.clone())]);
        assert_eq!(g.num_vars(), 1);

        // Verify by comparing with sequential partial evaluations
        for y in [FE::zero(), FE::one()] {
            let g_eval = g
                .evaluate(vec![y.clone()])
                .expect("Valid evaluation point for g");
            let f_eval = poly
                .evaluate(vec![r1.clone(), y.clone(), r2.clone()])
                .expect("Valid evaluation point for f");
            assert_eq!(g_eval, f_eval);
        }
    }

    #[test]
    fn partial_evaluate_many_empty() {
        let poly = DenseMultilinearPolynomial::new(vec![FE::from(1), FE::from(2)]);
        let result = poly.partial_evaluate_many(&[]);
        assert_eq!(result, poly);
    }

    #[test]
    fn batch_evaluate_correctness() {
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
        ]);

        let points = vec![
            vec![FE::zero(), FE::zero()],
            vec![FE::zero(), FE::one()],
            vec![FE::one(), FE::zero()],
            vec![FE::one(), FE::one()],
            vec![FE::from(5), FE::from(7)],
        ];

        let batch_results = poly
            .batch_evaluate(&points)
            .expect("All points have valid dimension");

        // Verify each result matches individual evaluation
        for (i, point) in points.iter().enumerate() {
            let individual = poly
                .evaluate(point.clone())
                .expect("Valid evaluation point");
            assert_eq!(batch_results[i], individual);
        }
    }

    #[test]
    fn batch_evaluate_empty() {
        let poly = DenseMultilinearPolynomial::new(vec![FE::from(1), FE::from(2)]);
        let results = poly
            .batch_evaluate(&[])
            .expect("Empty batch should succeed");
        assert!(results.is_empty());
    }

    #[test]
    fn batch_evaluate_wrong_dimension() {
        let poly = DenseMultilinearPolynomial::new(vec![FE::from(1), FE::from(2)]);
        let points = vec![
            vec![FE::one(), FE::one()], // Wrong: 2 vars instead of 1
        ];
        assert!(poly.batch_evaluate(&points).is_err());
    }

    #[test]
    fn tensor_product_basic() {
        // f(x) = [1, 2] (1 variable)
        let f = DenseMultilinearPolynomial::new(vec![FE::from(1), FE::from(2)]);
        // g(y) = [3, 4] (1 variable)
        let g = DenseMultilinearPolynomial::new(vec![FE::from(3), FE::from(4)]);

        // h(x, y) = f(x) * g(y) should have evals:
        // [f(0)*g(0), f(0)*g(1), f(1)*g(0), f(1)*g(1)] = [1*3, 1*4, 2*3, 2*4] = [3, 4, 6, 8]
        let h = f.tensor_product(&g);

        assert_eq!(h.num_vars(), 2);
        assert_eq!(h.len(), 4);
        assert_eq!(
            *h.evals(),
            vec![FE::from(3), FE::from(4), FE::from(6), FE::from(8)]
        );
    }

    #[test]
    fn tensor_product_correctness() {
        let f = DenseMultilinearPolynomial::new(vec![FE::from(1), FE::from(2)]);
        let g = DenseMultilinearPolynomial::new(vec![
            FE::from(3),
            FE::from(4),
            FE::from(5),
            FE::from(6),
        ]);

        let h = f.tensor_product(&g);
        assert_eq!(h.num_vars(), 3); // 1 + 2 = 3

        // Verify h(x, y1, y2) = f(x) * g(y1, y2) at some points
        for x in [FE::zero(), FE::one()] {
            for y1 in [FE::zero(), FE::one()] {
                for y2 in [FE::zero(), FE::one()] {
                    let h_eval = h
                        .evaluate(vec![x.clone(), y1.clone(), y2.clone()])
                        .expect("Valid point for h");
                    let f_eval = f.evaluate(vec![x.clone()]).expect("Valid point for f");
                    let g_eval = g
                        .evaluate(vec![y1.clone(), y2.clone()])
                        .expect("Valid point for g");
                    assert_eq!(h_eval, f_eval * g_eval);
                }
            }
        }
    }

    #[test]
    fn eq_polynomial_on_boolean_hypercube() {
        // eq(x, r) should be 1 at r and 0 at other boolean points
        let r = vec![FE::one(), FE::zero()]; // r = (1, 0)
        let eq_poly = eq_polynomial::<F>(&r);

        // Evaluate at all boolean points
        // (0, 0) -> 0
        // (0, 1) -> 0
        // (1, 0) -> 1
        // (1, 1) -> 0
        assert_eq!(
            eq_poly
                .evaluate(vec![FE::zero(), FE::zero()])
                .expect("Valid point"),
            FE::zero()
        );
        assert_eq!(
            eq_poly
                .evaluate(vec![FE::zero(), FE::one()])
                .expect("Valid point"),
            FE::zero()
        );
        assert_eq!(
            eq_poly
                .evaluate(vec![FE::one(), FE::zero()])
                .expect("Valid point"),
            FE::one()
        );
        assert_eq!(
            eq_poly
                .evaluate(vec![FE::one(), FE::one()])
                .expect("Valid point"),
            FE::zero()
        );
    }

    #[test]
    fn eq_polynomial_sum_is_one() {
        // sum_{w in {0,1}^n} eq(w, r) = 1 for any r
        let r = vec![FE::from(5), FE::from(7), FE::from(11)];
        let eq_poly = eq_polynomial::<F>(&r);

        // Sum all evaluations on the boolean hypercube
        let sum: FE = eq_poly.evals().iter().cloned().sum();
        assert_eq!(sum, FE::one());
    }

    #[test]
    fn eq_eval_matches_eq_polynomial() {
        let r = vec![FE::from(3), FE::from(7)];
        let eq_poly = eq_polynomial::<F>(&r);

        // Test at various points
        let test_points = vec![
            vec![FE::zero(), FE::zero()],
            vec![FE::one(), FE::zero()],
            vec![FE::from(5), FE::from(11)],
            vec![FE::from(13), FE::from(17)],
        ];

        for x in test_points {
            let via_poly = eq_poly.evaluate(x.clone()).expect("Valid point");
            let via_eval = eq_eval(&x, &r);
            assert_eq!(via_poly, via_eval);
        }
    }

    #[test]
    fn eq_eval_same_point_is_one() {
        // eq(r, r) should equal 1 for boolean r
        for r0 in [0u64, 1] {
            for r1 in [0u64, 1] {
                let r = vec![FE::from(r0), FE::from(r1)];
                let result = eq_eval(&r, &r);
                assert_eq!(result, FE::one());
            }
        }
    }

    #[test]
    fn eq_eval_different_boolean_points_is_zero() {
        // eq(x, r) = 0 when x != r for boolean x, r
        let test_cases = vec![
            (vec![0u64, 0], vec![0u64, 1]),
            (vec![0, 0], vec![1, 0]),
            (vec![0, 0], vec![1, 1]),
            (vec![1, 1], vec![0, 0]),
        ];

        for (x_vals, r_vals) in test_cases {
            let x: Vec<FE> = x_vals.iter().map(|&v| FE::from(v)).collect();
            let r: Vec<FE> = r_vals.iter().map(|&v| FE::from(v)).collect();
            let result = eq_eval(&x, &r);
            assert_eq!(result, FE::zero());
        }
    }

    #[test]
    fn from_evaluations_roundtrip() {
        let original_evals = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];
        let poly = DenseMultilinearPolynomial::from_evaluations(original_evals.clone());

        // Verify evaluations match on boolean hypercube
        let recovered = poly.to_evaluations();
        assert_eq!(recovered, original_evals);

        // Verify evaluation at each boolean point
        assert_eq!(
            poly.evaluate(vec![FE::zero(), FE::zero()])
                .expect("Valid point"),
            FE::from(1)
        );
        assert_eq!(
            poly.evaluate(vec![FE::zero(), FE::one()])
                .expect("Valid point"),
            FE::from(2)
        );
        assert_eq!(
            poly.evaluate(vec![FE::one(), FE::zero()])
                .expect("Valid point"),
            FE::from(3)
        );
        assert_eq!(
            poly.evaluate(vec![FE::one(), FE::one()])
                .expect("Valid point"),
            FE::from(4)
        );
    }

    #[test]
    fn partial_evaluate_consistency_with_full_evaluate() {
        // For any polynomial f and any point (a, b, c), we should have:
        // f.partial_evaluate(0, a).partial_evaluate(0, b).partial_evaluate(0, c).evals()[0]
        // == f.evaluate([a, b, c])

        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
            FE::from(5),
            FE::from(6),
            FE::from(7),
            FE::from(8),
        ]);

        let a = FE::from(13);
        let b = FE::from(17);
        let c = FE::from(23);

        // Evaluate directly
        let direct = poly
            .evaluate(vec![a.clone(), b.clone(), c.clone()])
            .expect("Valid point");

        // Evaluate via sequential partial evaluation
        let step1 = poly.partial_evaluate(0, &a);
        let step2 = step1.partial_evaluate(0, &b);
        let step3 = step2.partial_evaluate(0, &c);

        assert_eq!(step3.num_vars(), 0);
        assert_eq!(step3.evals()[0], direct);
    }
}
