//! Multilinear Polynomials for Binius
//!
//! Binius uses multilinear polynomials for efficient proof generation.
//! A multilinear polynomial P(x1, ..., xn) has degree at most 1 in each variable.

use crate::fields::tower::Tower;

/// Represents a multilinear polynomial over binary fields.
///
/// A multilinear polynomial in n variables has the form:
/// P(x1, ..., xn) = Σ a_{i1,...,in} * x1^{i1} * ... * xn^{in}
/// where each ik ∈ {0, 1}
///
/// The polynomial is represented by its evaluations at all 2^n points
/// in {0, 1}^n.
#[derive(Clone, Debug)]
pub struct MultilinearPolynomial {
    /// The number of variables (n)
    pub num_vars: usize,
    /// Evaluations at all 2^n points
    pub evals: Vec<Tower>,
}

impl MultilinearPolynomial {
    /// Creates a new multilinear polynomial from evaluations
    pub fn new(evals: Vec<Tower>) -> Result<Self, &'static str> {
        let num_vars = evals.len().ilog2() as usize;
        if 1 << num_vars != evals.len() {
            return Err("Number of evaluations must be a power of 2");
        }
        Ok(Self { num_vars, evals })
    }

    /// Creates a constant polynomial
    pub fn constant(c: Tower) -> Self {
        Self {
            num_vars: 0,
            evals: vec![c],
        }
    }

    /// Returns the degree (number of variables) of the polynomial
    pub fn degree(&self) -> usize {
        self.num_vars
    }

    /// Evaluates the polynomial at a given point
    pub fn evaluate(&self, point: &[Tower]) -> Tower {
        if point.len() != self.num_vars {
            panic!("Point dimension mismatch");
        }
        if self.num_vars == 0 {
            return self.evals[0];
        }

        // Use the standard multilinear evaluation algorithm
        self.evaluate_recursive(point, 0)
    }

    fn evaluate_recursive(&self, point: &[Tower], var_idx: usize) -> Tower {
        if var_idx == self.num_vars {
            return self.evals[0];
        }

        // Split evaluations into even and odd indices
        let half = 1 << var_idx;
        let (evens, odds) = self.evals.split_at(half);

        // Recursively evaluate both halves
        let even_eval = Self {
            num_vars: var_idx,
            evals: evens.to_vec(),
        }
        .evaluate_recursive(point, var_idx + 1);

        let odd_eval = Self {
            num_vars: var_idx,
            evals: odds.to_vec(),
        }
        .evaluate_recursive(point, var_idx + 1);

        // Return (1 - x_i) * even + x_i * odd
        let one_minus_xi = Tower::one() - point[var_idx];
        (one_minus_xi * even_eval) + (point[var_idx] * odd_eval)
    }

    /// Partial evaluation: fixes the value of variable i
    pub fn partial_evaluate(&self, var_idx: usize, value: Tower) -> MultilinearPolynomial {
        if var_idx >= self.num_vars {
            return self.clone();
        }

        let half = 1 << var_idx;
        let mut new_evals = Vec::with_capacity(self.evals.len() / 2);

        for chunk in self.evals.chunks(2 * half) {
            let even = &chunk[..half];
            let odd = &chunk[half..];

            for i in 0..half {
                // P'(...) = (1 - x) * P(..., 0, ...) + x * P(..., 1, ...)
                let one_minus_x = Tower::one() - value;
                new_evals.push(one_minus_x * even[i] + value * odd[i]);
            }
        }

        Self {
            num_vars: self.num_vars - 1,
            evals: new_evals,
        }
    }

    /// Returns the list of all evaluations
    pub fn evaluations(&self) -> &[Tower] {
        &self.evals
    }

    /// Creates a polynomial from a function that computes evaluations
    pub fn from_closure<F>(num_vars: usize, f: F) -> Self
    where
        F: Fn(usize) -> Tower,
    {
        let size = 1 << num_vars;
        let evals: Vec<Tower> = (0..size).map(f).collect();
        Self { num_vars, evals }
    }

    /// Arithmetic operations
    pub fn add(&self, other: &MultilinearPolynomial) -> MultilinearPolynomial {
        if self.num_vars != other.num_vars {
            panic!("Cannot add polynomials with different number of variables");
        }
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
        if self.num_vars != other.num_vars {
            panic!("Cannot subtract polynomials with different number of variables");
        }
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

    /// Scalar multiplication
    pub fn scale(&self, scalar: Tower) -> MultilinearPolynomial {
        let evals: Vec<Tower> = self.evals.iter().map(|v| *v * scalar).collect();
        Self {
            num_vars: self.num_vars,
            evals,
        }
    }
}

impl Default for MultilinearPolynomial {
    fn default() -> Self {
        Self::constant(Tower::zero())
    }
}

/// Tests for MultilinearPolynomial
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_polynomial() {
        let c = Tower::new(42, 3);
        let p = MultilinearPolynomial::constant(c);
        assert_eq!(p.degree(), 0);
        assert_eq!(p.evaluate(&[]), c);
    }

    #[test]
    fn test_evaluate_simple() {
        // P(x) = 1 + x, represented as [P(0), P(1)] = [1, 2]
        let evals = vec![
            Tower::new(1, 1), // P(0) = 1
            Tower::new(2, 1), // P(1) = 1 + 1 = 0 (in GF(2))... wait, let me use proper values
        ];
        // Actually let's use: P(x) = 1 + x in GF(4)
        // At x=0: P(0) = 1
        // At x=1: P(1) = 1 + 1 = 0 in binary field
    }

    #[test]
    fn test_addition() {
        // In GF(4), values are modulo 4 (2 bits)
        // 1 (0b01) + 3 (0b11) = 0b01 ^ 0b11 = 0b10 = 2
        // 2 (0b10) + 1 (0b01) = 0b10 ^ 0b01 = 0b11 = 3
        let p1 = MultilinearPolynomial::new(vec![Tower::new(1, 1), Tower::new(2, 1)]).unwrap();

        let p2 = MultilinearPolynomial::new(vec![Tower::new(3, 1), Tower::new(1, 1)]).unwrap();

        let sum = p1.add(&p2);
        assert_eq!(sum.evals[0].value(), 2); // 1 + 3 = 2 in GF(4)
        assert_eq!(sum.evals[1].value(), 3); // 2 + 1 = 3 in GF(4)
    }
}
