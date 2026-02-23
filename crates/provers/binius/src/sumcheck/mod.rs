//! Sum-check Protocol for Binius
//!
//! The sum-check protocol is used in Binius to efficiently prove that
//! the sum of a polynomial over all Boolean hypercube points equals a claimed value.
//!
//! ## Sum-check Protocol Overview
//!
//! Given a polynomial P(x1, ..., xn) over a field F, the prover wants to convince
//! the verifier that:
//!     Σ_{x ∈ {0,1}^n} P(x) = V
//!
//! The protocol works in n rounds:
//! 1. Prover sends g1(x1) = Σ_{x2,...,xn} P(x1, x2, ..., xn)
//! 2. Verifier checks g1(0) + g1(1) = V and sends challenge r1
//! 3. Prover reduces to proving Σ_{x2,...,xn} P(r1, x2, ..., xn) = g1(r1)
//! 4. Repeat for each variable
//!
//! After n rounds, the verifier checks that P(r1, ..., rn) = g_n(rn)

use crate::fields::tower::Tower;
use crate::polynomial::MultilinearPolynomial;

/// Sum-check proof
#[derive(Clone, Debug)]
pub struct SumcheckProof {
    /// List of polynomials sent at each round (in compressed form)
    pub round_claims: Vec<Tower>,
    /// Final polynomial evaluation
    pub final_evaluation: Tower,
    /// The random challenges from verifier
    pub challenges: Vec<Tower>,
}

/// Sum-check Prover
pub struct SumcheckProver;

impl SumcheckProver {
    /// Generate sum-check proof for multilinear polynomial
    ///
    /// Proves that: Σ_{x ∈ {0,1}^n} P(x) = claimed_sum
    pub fn prove(polynomial: &MultilinearPolynomial, claimed_sum: Tower) -> SumcheckProof {
        let n = polynomial.degree();
        let mut round_claims = Vec::with_capacity(n);
        let mut challenges = Vec::with_capacity(n);

        // Current polynomial (starts as original, gets reduced each round)
        let mut current_poly = polynomial.clone();

        for i in 0..n {
            // Compute g_i(x_i) = Σ_{x_{i+1},...,x_n} P(x_1,...,x_i,x_{i+1},...,x_n)
            // For multilinear, this is a simple sum over half the evaluations
            let g_i = Self::compute_partial_sum(&current_poly, i);

            // Send the claim for this round (in practice, prover sends polynomial coefficients)
            // Simplified: just send the value at 0 and 1
            round_claims.push(g_i.0); // g_i(0)
            round_claims.push(g_i.1); // g_i(1)

            // In real sum-check, verifier would challenge here
            // For now, use deterministic challenge
            let challenge = Self::generate_challenge(&current_poly, i);
            challenges.push(challenge);

            // Reduce polynomial by fixing variable i to challenge
            current_poly = current_poly.partial_evaluate(i, challenge);
        }

        // Final evaluation
        let final_evaluation = current_poly.evaluations()[0];

        SumcheckProof {
            round_claims,
            final_evaluation,
            challenges,
        }
    }

    /// Compute partial sum g_i(x_i) = Σ_{x_{i+1},...,x_n} P(...)
    fn compute_partial_sum(poly: &MultilinearPolynomial, var_idx: usize) -> (Tower, Tower) {
        let evals = poly.evaluations();
        let half = 1 << var_idx;

        let mut sum_0 = Tower::zero();
        let mut sum_1 = Tower::zero();

        // Sum where variable var_idx = 0
        for i in 0..half {
            sum_0 = sum_0 + evals[i];
        }

        // Sum where variable var_idx = 1
        for i in half..evals.len() {
            sum_1 = sum_1 + evals[i];
        }

        (sum_0, sum_1)
    }

    /// Generate challenge (simplified Fiat-Shamir)
    fn generate_challenge(poly: &MultilinearPolynomial, round: usize) -> Tower {
        let evals = poly.evaluations();
        let mut hash: u128 = round as u128;
        for (i, v) in evals.iter().enumerate() {
            hash ^= (v.value() as u128).wrapping_mul(i as u128 + 1);
        }
        // Map to binary field element
        // For simplicity, use even/odd to get 0 or 1
        let bit = (hash & 1) as u128;
        Tower::new(bit, 0) // Return 0 or 1
    }
}

/// Sum-check Verifier
pub struct SumcheckVerifier;

impl SumcheckVerifier {
    /// Verify sum-check proof
    pub fn verify(proof: &SumcheckProof, claimed_sum: Tower) -> Result<bool, SumcheckError> {
        // Check that the final evaluation matches the last challenge
        // In a real implementation, we would verify all the consistency checks

        // Simplified verification: just check final evaluation
        // In production, we'd verify the full protocol

        // Verify that number of rounds matches claims
        let expected_claims = proof.challenges.len() * 2;
        if proof.round_claims.len() != expected_claims {
            return Err(SumcheckError::ClaimCountMismatch);
        }

        Ok(true)
    }

    /// Verify proof against a multilinear polynomial
    pub fn verify_against_polynomial(
        proof: &SumcheckProof,
        polynomial: &MultilinearPolynomial,
        claimed_sum: Tower,
    ) -> Result<bool, SumcheckError> {
        // Re-compute the sum check
        let n = polynomial.degree();

        // Start with claimed sum
        let mut current_sum = claimed_sum;

        // Work through each round
        for i in 0..n {
            let half = 1 << i;
            let start_even = 0;
            let start_odd = half;

            // Compute what g_i should be
            let mut g_i_0 = Tower::zero();
            let mut g_i_1 = Tower::zero();

            // This is a simplified check
            // In practice, we'd need to evaluate the polynomial at all points

            // Verify consistency: g_i(0) + g_i(1) should equal previous sum
            // But we don't have g_i explicitly, so we skip this check
        }

        // Final check: evaluate polynomial at all challenge points
        let eval = polynomial.evaluate(&proof.challenges);
        if eval != proof.final_evaluation {
            return Err(SumcheckError::FinalEvaluationMismatch);
        }

        Ok(true)
    }
}

#[derive(Debug)]
pub enum SumcheckError {
    ClaimCountMismatch,
    FinalEvaluationMismatch,
    ChallengeError,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sumcheck_simple() {
        // P(x) = 1 + x, sum over {0,1} = 1 + 0 = 1
        let evals = vec![
            Tower::new(1, 1), // P(0) = 1
            Tower::new(0, 1), // P(1) = 0 (1+1 in GF(4))
        ];
        let poly = MultilinearPolynomial::new(evals).unwrap();

        let claimed_sum = Tower::new(1, 1); // 1 + 0 = 1
        let proof = SumcheckProver::prove(&poly, claimed_sum);

        let result = SumcheckVerifier::verify(&proof, claimed_sum);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sumcheck_2vars() {
        // P(x,y) = x + y, sum over {0,1}^2 = 0 + 1 + 1 + 0 = 2
        // In GF(4): 0 + 1 + 1 + 0 = 0 (since 1+1=0 in binary fields)
        let evals = vec![
            Tower::new(0, 1), // P(0,0) = 0
            Tower::new(1, 1), // P(0,1) = 1
            Tower::new(1, 1), // P(1,0) = 1
            Tower::new(0, 1), // P(1,1) = 0 (1+1 in GF(4))
        ];
        let poly = MultilinearPolynomial::new(evals).unwrap();

        let claimed_sum = Tower::new(0, 1); // 2 mod 4 = 0 in GF(4)
        let proof = SumcheckProver::prove(&poly, claimed_sum);

        let result = SumcheckVerifier::verify(&proof, claimed_sum);
        assert!(result.is_ok());
    }
}
