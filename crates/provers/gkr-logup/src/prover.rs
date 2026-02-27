use core::ops::Mul;

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{HasDefaultTranscript, IsField};
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::ByteConversion;

use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;

use lambdaworks_sumcheck::common::run_sumcheck_with_transcript;
use lambdaworks_sumcheck::common::SumcheckProver;
use lambdaworks_sumcheck::ProverError;

use lambdaworks_math::polynomial::DenseMultilinearPolynomial;

use crate::eq_evals::EqEvaluations;
use crate::fraction::Fraction;
use crate::layer::{gen_layers, Layer};
use crate::MAX_DEGREE;
use lambdaworks_math::polynomial::eq_eval;

use crate::utils::random_linear_combination;
use crate::verifier::{LayerMask, Proof, SumcheckProof, VerificationResult};

/// Multivariate polynomial oracle for the GKR protocol.
///
/// Represents `P(x) = eq_eval(x, y) * gate_output(x)` where gate_output depends on the layer type:
/// - GrandProduct: `inp(x, 0) * inp(x, 1)`
/// - LogUp: `(num(x,0)*den(x,1) + num(x,1)*den(x,0)) + lambda * den(x,0)*den(x,1)`
pub struct LayerOracle<F: IsField>
where
    F::BaseType: Send + Sync,
{
    eq_evals: EqEvaluations<F>,
    input_layer: Layer<F>,
    eq_fixed_var_correction: FieldElement<F>,
    lambda: FieldElement<F>,
}

impl<F: IsField> LayerOracle<F>
where
    F::BaseType: Send + Sync,
{
    /// Creates a new `LayerOracle`.
    ///
    /// Returns an error if `input_layer` has zero variables (i.e., is an output layer).
    fn new(
        eq_evals: EqEvaluations<F>,
        input_layer: Layer<F>,
        eq_fixed_var_correction: FieldElement<F>,
        lambda: FieldElement<F>,
    ) -> Result<Self, ProverError> {
        if input_layer.n_variables() == 0 {
            return Err(ProverError::InvalidState(
                "LayerOracle must not wrap an output layer (input_layer has 0 variables)".into(),
            ));
        }
        Ok(Self {
            eq_evals,
            input_layer,
            eq_fixed_var_correction,
            lambda,
        })
    }

    fn is_constant(&self) -> bool {
        self.n_variables() == 0
    }

    /// Number of remaining variables.
    pub fn n_variables(&self) -> usize {
        self.input_layer.n_variables() - 1
    }

    /// Computes the univariate polynomial `f(t) = sum_x g(t, x)` for all `x` in `{0,1}^(n-1)`.
    ///
    /// `claim` equals `f(0) + f(1)`, which can be used to derive `f(1)` from `f(0)`.
    pub fn sum_as_poly_in_first_variable(
        &self,
        claim: &FieldElement<F>,
    ) -> Result<Polynomial<FieldElement<F>>, ProverError> {
        let n_variables = self.n_variables();
        if n_variables == 0 {
            return Err(ProverError::InvalidState(
                "Cannot sum a constant oracle (0 variables)".into(),
            ));
        }
        let n_terms = 1 << (n_variables - 1);
        let y = self.eq_evals.y();
        let lambda = &self.lambda;

        let (mut eval_at_0, mut eval_at_2) = match &self.input_layer {
            Layer::GrandProduct(col) => eval_grand_product_sum(&self.eq_evals, col, n_terms),
            Layer::LogUpGeneric {
                numerators,
                denominators,
            }
            | Layer::LogUpMultiplicities {
                numerators,
                denominators,
            } => eval_logup_sum(
                &self.eq_evals,
                Some(numerators),
                denominators,
                n_terms,
                lambda,
            ),
            Layer::LogUpSingles { denominators } => {
                eval_logup_sum(&self.eq_evals, None, denominators, n_terms, lambda)
            }
        };

        eval_at_0 = &eval_at_0 * &self.eq_fixed_var_correction;
        eval_at_2 = &eval_at_2 * &self.eq_fixed_var_correction;

        correct_sum_as_poly_in_first_variable(eval_at_0, eval_at_2, claim, y, n_variables)
    }

    /// Fixes the first variable to `challenge`, returning a new oracle with one fewer variable.
    pub fn fix_first_variable(self, challenge: &FieldElement<F>) -> Self {
        if self.is_constant() {
            return self;
        }

        let y = self.eq_evals.y();
        let z0 = &y[y.len() - self.n_variables()];
        let eq_fixed_var_correction = &self.eq_fixed_var_correction
            * &eq_eval(std::slice::from_ref(challenge), std::slice::from_ref(z0));

        Self {
            eq_evals: self.eq_evals,
            eq_fixed_var_correction,
            input_layer: self.input_layer.fix_first_variable(challenge),
            lambda: self.lambda,
        }
    }

    /// Extracts the mask (column values at the two endpoints) from a constant oracle.
    pub fn try_into_mask(self) -> Option<LayerMask<F>> {
        if !self.is_constant() {
            return None;
        }

        let columns = match self.input_layer {
            Layer::GrandProduct(mle) => {
                vec![[mle[0].clone(), mle[1].clone()]]
            }
            Layer::LogUpGeneric {
                numerators,
                denominators,
            }
            | Layer::LogUpMultiplicities {
                numerators,
                denominators,
            } => {
                vec![
                    [numerators[0].clone(), numerators[1].clone()],
                    [denominators[0].clone(), denominators[1].clone()],
                ]
            }
            Layer::LogUpSingles { denominators } => {
                vec![
                    [FieldElement::one(), FieldElement::one()],
                    [denominators[0].clone(), denominators[1].clone()],
                ]
            }
        };

        Some(LayerMask::new(columns))
    }
}

/// Evaluates `sum_x eq_eval((0^|r|, 0, x), y) * inp(r, t, x, 0) * inp(r, t, x, 1)` at t=0 and t=2.
fn eval_grand_product_sum<F: IsField>(
    eq_evals: &EqEvaluations<F>,
    input_layer: &DenseMultilinearPolynomial<F>,
    n_terms: usize,
) -> (FieldElement<F>, FieldElement<F>)
where
    F::BaseType: Send + Sync,
{
    let mut eval_at_0 = FieldElement::<F>::zero();
    let mut eval_at_2 = FieldElement::<F>::zero();

    for i in 0..n_terms {
        // Input polynomial at points (r, {0, 1, 2}, bits(i), {0, 1})
        let inp_at_r0i0 = &input_layer[i * 2];
        let inp_at_r0i1 = &input_layer[i * 2 + 1];
        let inp_at_r1i0 = &input_layer[(n_terms + i) * 2];
        let inp_at_r1i1 = &input_layer[(n_terms + i) * 2 + 1];
        // inp(r, 2, x) = 2 * inp(r, 1, x) - inp(r, 0, x)
        let inp_at_r2i0 = &(inp_at_r1i0 + inp_at_r1i0) - inp_at_r0i0;
        let inp_at_r2i1 = &(inp_at_r1i1 + inp_at_r1i1) - inp_at_r0i1;

        // Product polynomial: prod(x) = inp(x, 0) * inp(x, 1) at (r, {0, 2}, bits(i))
        let prod_at_r0i = inp_at_r0i0 * inp_at_r0i1;
        let prod_at_r2i = &inp_at_r2i0 * &inp_at_r2i1;

        let eq_eval = &eq_evals[i];
        eval_at_0 = &eval_at_0 + &(eq_eval * &prod_at_r0i);
        eval_at_2 = &eval_at_2 + &(eq_eval * &prod_at_r2i);
    }

    (eval_at_0, eval_at_2)
}

/// Evaluates the LogUp sum at t=0 and t=2 for layers with explicit numerators.
fn eval_logup_sum<F: IsField>(
    eq_evals: &EqEvaluations<F>,
    input_numerators: Option<&DenseMultilinearPolynomial<F>>,
    input_denominators: &DenseMultilinearPolynomial<F>,
    n_terms: usize,
    lambda: &FieldElement<F>,
) -> (FieldElement<F>, FieldElement<F>)
where
    F::BaseType: Send + Sync,
{
    let mut eval_at_0 = FieldElement::<F>::zero();
    let mut eval_at_2 = FieldElement::<F>::zero();

    let get_num = |idx: usize| -> FieldElement<F> {
        match input_numerators {
            Some(nums) => nums[idx].clone(),
            None => FieldElement::one(),
        }
    };

    for i in 0..n_terms {
        let num_r0i0 = get_num(i * 2);
        let den_r0i0 = &input_denominators[i * 2];
        let num_r0i1 = get_num(i * 2 + 1);
        let den_r0i1 = &input_denominators[i * 2 + 1];
        let num_r1i0 = get_num((n_terms + i) * 2);
        let den_r1i0 = &input_denominators[(n_terms + i) * 2];
        let num_r1i1 = get_num((n_terms + i) * 2 + 1);
        let den_r1i1 = &input_denominators[(n_terms + i) * 2 + 1];

        // Extrapolate to t=2: val(r, 2, x) = 2 * val(r, 1, x) - val(r, 0, x)
        let num_r2i0 = &(&num_r1i0 + &num_r1i0) - &num_r0i0;
        let den_r2i0 = &(den_r1i0 + den_r1i0) - den_r0i0;
        let num_r2i1 = &(&num_r1i1 + &num_r1i1) - &num_r0i1;
        let den_r2i1 = &(den_r1i1 + den_r1i1) - den_r0i1;

        // Fraction addition at t=0
        let frac_r0 =
            Fraction::new(num_r0i0, den_r0i0.clone()) + Fraction::new(num_r0i1, den_r0i1.clone());
        // Fraction addition at t=2
        let frac_r2 = Fraction::new(num_r2i0, den_r2i0) + Fraction::new(num_r2i1, den_r2i1);

        let eq_eval = &eq_evals[i];
        eval_at_0 =
            &eval_at_0 + &(eq_eval * &(&frac_r0.numerator + &(lambda * &frac_r0.denominator)));
        eval_at_2 =
            &eval_at_2 + &(eq_eval * &(&frac_r2.numerator + &(lambda * &frac_r2.denominator)));
    }

    (eval_at_0, eval_at_2)
}

/// Computes `r(t) = sum_x eq_eval((t, x), y[-k:]) * p(t, x)` from evaluations of
/// `f(t) = sum_x eq_eval(({0}^(n - k), 0, x), y) * p(t, x)`.
///
/// Uses 4-point Lagrange interpolation at {0, 1, 2, b} where b is the root
/// of eq_eval(t, y[n-k]).
///
/// See <https://ia.cr/2024/108> (section 3.2).
fn correct_sum_as_poly_in_first_variable<F: IsField>(
    f_at_0: FieldElement<F>,
    f_at_2: FieldElement<F>,
    claim: &FieldElement<F>,
    y: &[FieldElement<F>],
    k: usize,
) -> Result<Polynomial<FieldElement<F>>, ProverError> {
    assert!(k > 0);
    let n = y.len();
    assert!(k <= n);

    // a_const = 1 / eq_eval((0^(n-k+1)), y[..n-k+1])
    // eq_eval(0, y_i) = 1 - y_i, so eq_eval(0^m, y) = prod(1 - y_i).
    // This is zero when any challenge y_i = 1, probability ~(n-k+1)/p.
    let a_const = y[..n - k + 1]
        .iter()
        .map(|yi| FieldElement::<F>::one() - yi)
        .fold(FieldElement::<F>::one(), |acc, v| acc * v)
        .inv()
        .map_err(|_| {
            ProverError::InvalidState(
                "eq_eval correction factor is zero (Fiat-Shamir challenge y_i = 1)".to_string(),
            )
        })?;

    // Find the root of eq_eval(t, y[n-k]):
    //   eq_eval(t, y[n-k]) = t * y[n-k] + (1-t)*(1-y[n-k])
    //   0 = 1 - y[n-k] - t*(1 - 2*y[n-k])
    //   t = (1 - y[n-k]) / (1 - 2*y[n-k])
    let one = FieldElement::<F>::one();
    let two = FieldElement::<F>::from(2u64);
    let y_nk = &y[n - k];
    let b_const = (&one - y_nk)
        * (&one - &(&two * y_nk)).inv().map_err(|_| {
            ProverError::InvalidState(
                "1 - 2*y_{n-k} is zero (Fiat-Shamir challenge y_{n-k} = 1/2)".to_string(),
            )
        })?;

    // r(0) = f(0) * eq_eval(0, y[n-k]) * a_const
    let zero = FieldElement::<F>::zero();
    let eq_at_0 = eq_eval(std::slice::from_ref(&zero), std::slice::from_ref(y_nk));
    let r_at_0 = &f_at_0 * &eq_at_0 * &a_const;

    // r(1) = claim - r(0) (since claim = r(0) + r(1))
    let r_at_1 = claim - &r_at_0;

    // r(2) = f(2) * eq_eval(2, y[n-k]) * a_const
    let eq_at_2 = eq_eval(std::slice::from_ref(&two), std::slice::from_ref(y_nk));
    let r_at_2 = &f_at_2 * &eq_at_2 * &a_const;

    // Interpolate degree-3 polynomial through (0, r(0)), (1, r(1)), (2, r(2)), (b, 0)
    // r(b) = 0 because b is the root of eq_eval(t, y[n-k]).
    Ok(Polynomial::interpolate(
        &[FieldElement::zero(), one, two, b_const],
        &[r_at_0, r_at_1, r_at_2, FieldElement::zero()],
    )?)
}

/// Adapter that wraps a [`LayerOracle`] to implement the sumcheck crate's
/// [`SumcheckProver`] trait, allowing the GKR prover to use
/// [`run_sumcheck_with_transcript`] for its per-layer sumcheck.
struct LayerSumcheckProver<F: IsField>
where
    F::BaseType: Send + Sync,
{
    oracle: Option<LayerOracle<F>>,
    claim: FieldElement<F>,
    last_round_poly: Option<Polynomial<FieldElement<F>>>,
}

impl<F: IsField> LayerSumcheckProver<F>
where
    F::BaseType: Send + Sync,
{
    fn new(oracle: LayerOracle<F>, claim: FieldElement<F>) -> Self {
        Self {
            oracle: Some(oracle),
            claim,
            last_round_poly: None,
        }
    }

    /// Extracts the oracle, applying the final challenge to make it constant.
    ///
    /// During `run_sumcheck_with_transcript`, the adapter fixes one variable per round
    /// except the first, so the last challenge needs to be applied here.
    fn into_oracle(
        self,
        final_challenge: Option<&FieldElement<F>>,
    ) -> Result<LayerOracle<F>, ProverError> {
        let mut oracle = self.oracle.ok_or_else(|| {
            ProverError::InvalidState("oracle must be present after sumcheck".into())
        })?;
        if let Some(r) = final_challenge {
            oracle = oracle.fix_first_variable(r);
        }
        Ok(oracle)
    }
}

impl<F: IsField> SumcheckProver<F> for LayerSumcheckProver<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    fn num_vars(&self) -> usize {
        // The oracle is always present during sumcheck rounds.
        // If it's missing, this is an internal invariant violation.
        self.oracle.as_ref().map_or(0, |o| o.n_variables())
    }

    fn num_factors(&self) -> usize {
        MAX_DEGREE
    }

    fn compute_initial_sum(&self) -> FieldElement<F> {
        self.claim.clone()
    }

    fn round(
        &mut self,
        r_prev: Option<&FieldElement<F>>,
    ) -> Result<Polynomial<FieldElement<F>>, ProverError> {
        if let Some(r) = r_prev {
            self.claim = self
                .last_round_poly
                .as_ref()
                .ok_or_else(|| {
                    ProverError::InvalidState(
                        "last_round_poly must be set after first round".into(),
                    )
                })?
                .evaluate(r);
            let oracle = self.oracle.take().ok_or_else(|| {
                ProverError::InvalidState("oracle must be present during sumcheck".into())
            })?;
            self.oracle = Some(oracle.fix_first_variable(r));
        }
        let poly = self
            .oracle
            .as_ref()
            .ok_or_else(|| {
                ProverError::InvalidState("oracle must be present during sumcheck".into())
            })?
            .sum_as_poly_in_first_variable(&self.claim)?;
        self.last_round_poly = Some(poly.clone());
        Ok(poly)
    }
}

/// Random linear combination of polynomials: `p_0 + alpha * p_1 + ... + alpha^(n-1) * p_{n-1}`.
fn poly_random_linear_combination<F: IsField>(
    polys: &[Polynomial<FieldElement<F>>],
    alpha: &FieldElement<F>,
) -> Polynomial<FieldElement<F>> {
    polys.iter().rev().fold(
        Polynomial::new(&[FieldElement::<F>::zero()]),
        |acc, poly| acc * Polynomial::new(std::slice::from_ref(alpha)) + poly.clone(),
    )
}

/// Proves a batch of sumcheck instances with a single shared proof, using
/// `LayerOracle` methods directly.
///
/// Combines multiple oracles into one via random linear combination with `alpha`.
/// Oracles can have different numbers of variables — smaller ones are scaled by a
/// "doubling factor" and produce a constant polynomial until their variables begin.
///
/// # Doubling factor correctness
///
/// When an oracle P has k variables but the batch sumcheck runs for n > k rounds,
/// we treat P as a polynomial P' over n variables that ignores the first n-k:
///
/// `P'(x_1, ..., x_n) = P(x_{n-k+1}, ..., x_n)`
///
/// Summing over the Boolean hypercube:
///
/// `Σ P'(x) = 2^{n-k} · Σ P(x)`    (over `x ∈ {0,1}^n` and `x ∈ {0,1}^k`)
///
/// because each of the `2^{n-k}` assignments to the unused variables contributes
/// the same inner sum. This is why we multiply the claimed sum by `2^{n-k}`.
///
/// During the first n-k rounds, P' does not depend on the current variable,
/// so its round polynomial is the constant `claim / 2`: evaluating at 0 and 1
/// both give `claim / 2`, and `claim/2 + claim/2 = claim` satisfies the
/// sumcheck relation. After n-k rounds the claim has been halved n-k times
/// back to the original sum, and the oracle begins its real computation.
///
/// Returns `(proof, assignment, constant_oracles, final_claims)`.
#[allow(clippy::type_complexity)]
fn prove_batch_sumcheck<F, T>(
    mut claims: Vec<FieldElement<F>>,
    mut oracles: Vec<LayerOracle<F>>,
    alpha: &FieldElement<F>,
    transcript: &mut T,
) -> Result<
    (
        SumcheckProof<F>,
        Vec<FieldElement<F>>,
        Vec<LayerOracle<F>>,
        Vec<FieldElement<F>>,
    ),
    ProverError,
>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
    T: IsTranscript<F>,
{
    if claims.len() != oracles.len() {
        return Err(ProverError::InvalidState(
            "claims and oracles length mismatch in prove_batch_sumcheck".into(),
        ));
    }
    let n_variables = oracles
        .iter()
        .map(|o| o.n_variables())
        .max()
        .ok_or_else(|| {
            ProverError::InvalidState("prove_batch_sumcheck requires at least one oracle".into())
        })?;

    let mut round_polys = Vec::with_capacity(n_variables);
    let mut assignment = Vec::with_capacity(n_variables);

    // Scale claims by doubling factor for smaller instances.
    let two_inv = FieldElement::<F>::from(2u64)
        .inv()
        .map_err(|_| ProverError::InvalidState("field has characteristic 2".into()))?;
    for (claim, oracle) in claims.iter_mut().zip(oracles.iter()) {
        let n_unused = n_variables - oracle.n_variables();
        if n_unused > 0 {
            *claim = &*claim * &FieldElement::<F>::from(1u64 << n_unused);
        }
    }

    for round in 0..n_variables {
        let n_remaining = n_variables - round;

        // Compute per-oracle round polynomials.
        let this_round_polys: Vec<Polynomial<FieldElement<F>>> = oracles
            .iter()
            .zip(claims.iter())
            .map(|(oracle, claim)| {
                if n_remaining == oracle.n_variables() {
                    oracle.sum_as_poly_in_first_variable(claim)
                } else {
                    // Oracle hasn't started yet: constant polynomial = claim / 2.
                    let half_claim = claim * &two_inv;
                    Ok(Polynomial::new(&[half_claim]))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Combine with alpha via random linear combination of polynomials.
        let combined = poly_random_linear_combination(&this_round_polys, alpha);

        // Sanity check.
        debug_assert_eq!(
            combined.evaluate(&FieldElement::<F>::zero())
                + combined.evaluate(&FieldElement::<F>::one()),
            random_linear_combination(&claims, alpha)
        );
        debug_assert!(combined.degree() <= MAX_DEGREE);

        // Send combined polynomial to verifier.
        for coeff in combined.coefficients() {
            transcript.append_field_element(coeff);
        }
        let challenge: FieldElement<F> = transcript.sample_field_element();

        // Update per-oracle claims.
        claims = this_round_polys
            .iter()
            .map(|p| p.evaluate(&challenge))
            .collect();

        // Fix first variable on active oracles.
        oracles = oracles
            .into_iter()
            .map(|oracle| {
                if n_remaining != oracle.n_variables() {
                    return oracle;
                }
                oracle.fix_first_variable(&challenge)
            })
            .collect();

        round_polys.push(combined);
        assignment.push(challenge);
    }

    let proof = SumcheckProof { round_polys };
    Ok((proof, assignment, oracles, claims))
}

/// Proves a single GKR instance.
///
/// The input layer should be committed to the transcript before calling this function.
///
/// Returns a `Proof` and a `VerificationResult` containing the OOD point and claims
/// to verify against the input layer's MLE.
pub fn prove<F, T>(
    transcript: &mut T,
    input_layer: Layer<F>,
) -> Result<(Proof<F>, VerificationResult<F>), ProverError>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
    T: IsTranscript<F>,
{
    // Generate all layers from input (leaves) to output (root)
    let layers = gen_layers(input_layer)?;
    let n_layers = layers.len() - 1; // number of transitions

    // Get output values from the root layer
    let output_claims = layers
        .last()
        .ok_or_else(|| ProverError::InvalidState("no layers generated".into()))?
        .try_into_output_layer_values()
        .ok_or_else(|| ProverError::InvalidState("last layer should be output".into()))?;

    // Append output claims to transcript
    for claim in &output_claims {
        transcript.append_field_element(claim);
    }

    // Sample lambda for combining columns
    let lambda: FieldElement<F> = transcript.sample_field_element();

    let mut sumcheck_proofs = Vec::new();
    let mut layer_masks = Vec::new();
    let mut ood_point: Vec<FieldElement<F>> = Vec::new();

    // Initial claim: random linear combination of output values
    let mut claims_to_verify = output_claims.clone();

    // Process layers from output to input (reverse order, skip the root)
    // layers[n_layers] is the root (output), layers[0] is the input
    // We iterate from layers[n_layers-1] down to layers[0]
    for layer_idx in (0..n_layers).rev() {
        let layer = &layers[layer_idx];

        // Generate eq_evals from current ood_point
        let eq_evals = EqEvaluations::generate(&ood_point);

        // Create the multivariate polynomial oracle
        let claim = random_linear_combination(&claims_to_verify, &lambda);

        let oracle =
            LayerOracle::new(eq_evals, layer.clone(), FieldElement::one(), lambda.clone())?;

        // Run sumcheck via the adapter
        let mut layer_prover = LayerSumcheckProver::new(oracle, claim);
        let (round_polys, sumcheck_ood_point) =
            run_sumcheck_with_transcript(&mut layer_prover, transcript)?;
        let constant_oracle = layer_prover.into_oracle(sumcheck_ood_point.last())?;
        let sumcheck_proof = SumcheckProof { round_polys };

        // Extract mask from the constant oracle
        let mask = constant_oracle.try_into_mask().ok_or_else(|| {
            ProverError::InvalidState("oracle is not constant after all sumcheck rounds".into())
        })?;

        // Append mask to transcript
        for col in mask.columns() {
            transcript.append_field_element(&col[0]);
            transcript.append_field_element(&col[1]);
        }

        // Sample challenge for next layer
        let challenge: FieldElement<F> = transcript.sample_field_element();

        // Reduce mask at challenge point (borrow challenge before moving it)
        claims_to_verify = mask.reduce_at_point(&challenge);

        // Update ood_point
        ood_point = sumcheck_ood_point;
        ood_point.push(challenge);

        sumcheck_proofs.push(sumcheck_proof);
        layer_masks.push(mask);
    }

    let proof = Proof {
        sumcheck_proofs,
        layer_masks,
        output_claims,
    };

    let artifact = VerificationResult {
        ood_point,
        claims_to_verify,
        n_variables: n_layers,
    };

    Ok((proof, artifact))
}

/// Proves multiple GKR instances simultaneously with a single shared sumcheck per layer.
///
/// Instances can have different numbers of variables (different sizes). Smaller instances
/// are scaled by a "doubling factor" so they participate in the same sumcheck round.
///
/// Returns a `BatchProof` and `BatchVerificationResult` containing the shared OOD point
/// and per-instance claims to verify against input layer MLEs.
pub fn prove_batch<F, T>(
    transcript: &mut T,
    input_layers: Vec<Layer<F>>,
) -> Result<
    (
        crate::verifier::BatchProof<F>,
        crate::verifier::BatchVerificationResult<F>,
    ),
    ProverError,
>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
    T: IsTranscript<F>,
{
    let n_instances = input_layers.len();

    // Domain separation: commit the batch protocol tag and instance count so the
    // transcript is structurally distinct from single-instance proofs and from
    // batches of different sizes (prevents proof malleability across modes).
    transcript.append_bytes(b"gkr_batch");
    transcript.append_bytes(&(n_instances as u64).to_le_bytes());

    if n_instances == 0 {
        return Ok((
            crate::verifier::BatchProof {
                sumcheck_proofs: Vec::new(),
                layer_masks_by_instance: Vec::new(),
                output_claims_by_instance: Vec::new(),
            },
            crate::verifier::BatchVerificationResult {
                ood_point: Vec::new(),
                claims_to_verify_by_instance: Vec::new(),
                n_variables_by_instance: Vec::new(),
            },
        ));
    }
    let n_layers_by_instance: Vec<usize> = input_layers.iter().map(|l| l.n_variables()).collect();
    let n_layers = *n_layers_by_instance
        .iter()
        .max()
        .ok_or_else(|| ProverError::InvalidState("prove_batch: no instances".into()))?;

    // Generate all layers per instance and reverse for output-to-input traversal.
    let mut layers_by_instance: Vec<std::iter::Rev<std::vec::IntoIter<Layer<F>>>> = input_layers
        .into_iter()
        .map(|input_layer| gen_layers(input_layer).map(|l| l.into_iter().rev()))
        .collect::<Result<Vec<_>, _>>()?;

    let mut output_claims_by_instance: Vec<Option<Vec<FieldElement<F>>>> = vec![None; n_instances];
    let mut layer_masks_by_instance: Vec<Vec<LayerMask<F>>> = vec![Vec::new(); n_instances];
    let mut sumcheck_proofs = Vec::new();

    let mut ood_point: Vec<FieldElement<F>> = Vec::new();
    let mut claims_to_verify_by_instance: Vec<Option<Vec<FieldElement<F>>>> =
        vec![None; n_instances];

    // Handle zero-layer instances (size-1 inputs: already at output, no sumcheck needed).
    for (instance, layers) in layers_by_instance.iter_mut().enumerate() {
        if n_layers_by_instance[instance] == 0 {
            let output_layer = layers.next().ok_or_else(|| {
                ProverError::InvalidState("zero-layer instance has no layers".into())
            })?;
            let output_values = output_layer.try_into_output_layer_values().ok_or_else(|| {
                ProverError::InvalidState("zero-layer instance: not an output layer".into())
            })?;
            claims_to_verify_by_instance[instance] = Some(output_values.clone());
            output_claims_by_instance[instance] = Some(output_values);
        }
    }

    for layer in 0..n_layers {
        let n_remaining_layers = n_layers - layer;

        // Detect output layers for each instance.
        for (instance, layers) in layers_by_instance.iter_mut().enumerate() {
            if n_layers_by_instance[instance] == n_remaining_layers {
                let output_layer = layers.next().ok_or_else(|| {
                    ProverError::InvalidState(format!(
                        "instance {instance}: no output layer available"
                    ))
                })?;
                let output_layer_values =
                    output_layer.try_into_output_layer_values().ok_or_else(|| {
                        ProverError::InvalidState(format!(
                            "instance {instance}: not an output layer"
                        ))
                    })?;
                claims_to_verify_by_instance[instance] = Some(output_layer_values.clone());
                output_claims_by_instance[instance] = Some(output_layer_values);
            }
        }

        // Seed transcript with active claims.
        for claims in claims_to_verify_by_instance.iter().flatten() {
            for claim in claims {
                transcript.append_field_element(claim);
            }
        }

        // Generate shared eq_evals and sample randomness.
        let eq_evals = EqEvaluations::generate(&ood_point);
        let sumcheck_alpha: FieldElement<F> = transcript.sample_field_element();
        let lambda: FieldElement<F> = transcript.sample_field_element();

        let mut sumcheck_oracles = Vec::new();
        let mut sumcheck_claims = Vec::new();
        let mut sumcheck_instances = Vec::new();

        // Create oracles for active instances (skip zero-layer instances).
        for (instance, claims) in claims_to_verify_by_instance.iter().enumerate() {
            if let Some(claims) = claims {
                if n_layers_by_instance[instance] == 0 {
                    continue;
                }
                let next_layer = layers_by_instance[instance].next().ok_or_else(|| {
                    ProverError::InvalidState(format!(
                        "instance {instance}: no next layer available"
                    ))
                })?;
                // TODO: eq_evals is read-only during sumcheck — share via Arc instead of cloning.
                let oracle = LayerOracle::new(
                    eq_evals.clone(),
                    next_layer,
                    FieldElement::one(),
                    lambda.clone(),
                )?;
                let claim = random_linear_combination(claims, &lambda);
                sumcheck_oracles.push(oracle);
                sumcheck_claims.push(claim);
                sumcheck_instances.push(instance);
            }
        }

        // Run batch sumcheck.
        let (sumcheck_proof, sumcheck_ood_point, constant_oracles, _final_claims) =
            prove_batch_sumcheck(
                sumcheck_claims,
                sumcheck_oracles,
                &sumcheck_alpha,
                transcript,
            )?;

        sumcheck_proofs.push(sumcheck_proof);

        // Extract masks from constant oracles and seed transcript.
        let masks: Vec<LayerMask<F>> = constant_oracles
            .into_iter()
            .map(|oracle| {
                oracle.try_into_mask().ok_or_else(|| {
                    ProverError::InvalidState("oracle should be constant after sumcheck".into())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        for (&instance, mask) in sumcheck_instances.iter().zip(masks.iter()) {
            for col in mask.columns() {
                transcript.append_field_element(&col[0]);
                transcript.append_field_element(&col[1]);
            }
            layer_masks_by_instance[instance].push(mask.clone());
        }

        // Sample challenge, reduce masks (borrow challenge before moving it).
        let challenge: FieldElement<F> = transcript.sample_field_element();
        for (instance, mask) in sumcheck_instances.into_iter().zip(masks) {
            claims_to_verify_by_instance[instance] = Some(mask.reduce_at_point(&challenge));
        }

        // Update OOD point.
        ood_point = sumcheck_ood_point;
        ood_point.push(challenge);
    }

    let output_claims_by_instance: Vec<Vec<FieldElement<F>>> = output_claims_by_instance
        .into_iter()
        .enumerate()
        .map(|(i, o)| {
            o.ok_or_else(|| {
                ProverError::InvalidState(format!("instance {i}: missing output claims"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let claims_to_verify_by_instance: Vec<Vec<FieldElement<F>>> = claims_to_verify_by_instance
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            c.ok_or_else(|| {
                ProverError::InvalidState(format!("instance {i}: missing claims to verify"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let proof = crate::verifier::BatchProof {
        sumcheck_proofs,
        layer_masks_by_instance,
        output_claims_by_instance,
    };

    let artifact = crate::verifier::BatchVerificationResult {
        ood_point,
        claims_to_verify_by_instance,
        n_variables_by_instance: n_layers_by_instance,
    };

    Ok((proof, artifact))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verifier::{self, Gate, LayerMask};
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    const MODULUS: u64 = 2013265921;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    /// Helper: prove and verify a GrandProduct layer, returning the artifact.
    fn prove_and_verify_grand_product(values: Vec<FE>) -> VerificationResult<F> {
        let input_layer = Layer::GrandProduct(DenseMultilinearPolynomial::new(values));

        let mut prover_transcript = DefaultTranscript::<F>::new(&[]);
        let (proof, _artifact) = prove(&mut prover_transcript, input_layer).unwrap();

        let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
        verifier::verify(Gate::GrandProduct, &proof, &mut verifier_transcript)
            .expect("verification should succeed")
    }

    /// Helper: prove and verify a LogUp layer, returning the artifact.
    fn prove_and_verify_logup(input_layer: Layer<F>) -> VerificationResult<F> {
        let mut prover_transcript = DefaultTranscript::<F>::new(&[]);
        let (proof, _artifact) = prove(&mut prover_transcript, input_layer).unwrap();

        let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
        verifier::verify(Gate::LogUp, &proof, &mut verifier_transcript)
            .expect("verification should succeed")
    }

    #[test]
    fn grand_product_prove() {
        let values: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let product: FE = values.iter().cloned().reduce(|a, b| a * b).unwrap();

        let input_layer = Layer::GrandProduct(DenseMultilinearPolynomial::new(values));
        let mut transcript = DefaultTranscript::<F>::new(&[]);
        let (proof, artifact) = prove(&mut transcript, input_layer).unwrap();

        assert_eq!(proof.output_claims, vec![product]);
        assert_eq!(proof.sumcheck_proofs.len(), 3); // log2(8) = 3 layers
        assert_eq!(proof.layer_masks.len(), 3);
        assert_eq!(artifact.ood_point.len(), 3);
    }

    #[test]
    fn logup_generic_prove() {
        let numerators: Vec<FE> = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];
        let denominators: Vec<FE> = vec![FE::from(5), FE::from(6), FE::from(7), FE::from(8)];

        let expected: Fraction<F> = numerators
            .iter()
            .zip(denominators.iter())
            .map(|(n, d)| Fraction::new(*n, *d))
            .sum();

        let input_layer = Layer::LogUpGeneric {
            numerators: DenseMultilinearPolynomial::new(numerators),
            denominators: DenseMultilinearPolynomial::new(denominators),
        };
        let mut transcript = DefaultTranscript::<F>::new(&[]);
        let (proof, _artifact) = prove(&mut transcript, input_layer).unwrap();

        assert_eq!(proof.output_claims.len(), 2);
        let out_ratio = proof.output_claims[0] * expected.denominator.inv().unwrap();
        let exp_ratio = expected.numerator * proof.output_claims[1].inv().unwrap();
        assert_eq!(out_ratio, exp_ratio);
    }

    // ---- End-to-end prove + verify roundtrip tests ----

    #[test]
    fn grand_product_prove_verify_roundtrip_size_4() {
        let values: Vec<FE> = (1u64..=4).map(FE::from).collect();
        let artifact = prove_and_verify_grand_product(values);
        assert_eq!(artifact.n_variables, 2);
        assert_eq!(artifact.ood_point.len(), 2);
        assert_eq!(artifact.claims_to_verify.len(), 1);
    }

    #[test]
    fn grand_product_prove_verify_roundtrip_size_8() {
        let values: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let artifact = prove_and_verify_grand_product(values);
        assert_eq!(artifact.n_variables, 3);
        assert_eq!(artifact.ood_point.len(), 3);
    }

    #[test]
    fn grand_product_prove_verify_roundtrip_size_16() {
        let values: Vec<FE> = (1u64..=16).map(FE::from).collect();
        let artifact = prove_and_verify_grand_product(values);
        assert_eq!(artifact.n_variables, 4);
    }

    #[test]
    fn logup_generic_prove_verify_roundtrip() {
        let numerators: Vec<FE> = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];
        let denominators: Vec<FE> = vec![FE::from(5), FE::from(6), FE::from(7), FE::from(8)];
        let input_layer = Layer::LogUpGeneric {
            numerators: DenseMultilinearPolynomial::new(numerators),
            denominators: DenseMultilinearPolynomial::new(denominators),
        };
        let artifact = prove_and_verify_logup(input_layer);
        assert_eq!(artifact.n_variables, 2);
        assert_eq!(artifact.claims_to_verify.len(), 2);
    }

    #[test]
    fn logup_singles_prove_verify_roundtrip() {
        let denominators: Vec<FE> = vec![FE::from(2), FE::from(3), FE::from(5), FE::from(7)];
        let input_layer = Layer::LogUpSingles {
            denominators: DenseMultilinearPolynomial::new(denominators),
        };
        let artifact = prove_and_verify_logup(input_layer);
        assert_eq!(artifact.n_variables, 2);
        assert_eq!(artifact.claims_to_verify.len(), 2);
    }

    #[test]
    fn logup_multiplicities_prove_verify_roundtrip() {
        let numerators: Vec<FE> = vec![FE::from(3), FE::from(1), FE::from(2), FE::from(1)];
        let denominators: Vec<FE> = vec![FE::from(10), FE::from(20), FE::from(30), FE::from(40)];
        let input_layer = Layer::LogUpMultiplicities {
            numerators: DenseMultilinearPolynomial::new(numerators),
            denominators: DenseMultilinearPolynomial::new(denominators),
        };
        let artifact = prove_and_verify_logup(input_layer);
        assert_eq!(artifact.n_variables, 2);
    }

    #[test]
    fn logup_singles_prove_verify_size_8() {
        let denominators: Vec<FE> = (2u64..=9).map(FE::from).collect();
        let input_layer = Layer::LogUpSingles {
            denominators: DenseMultilinearPolynomial::new(denominators),
        };
        let artifact = prove_and_verify_logup(input_layer);
        assert_eq!(artifact.n_variables, 3);
    }

    #[test]
    fn logup_generic_prove_verify_size_8() {
        let numerators: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let denominators: Vec<FE> = (11u64..=18).map(FE::from).collect();
        let input_layer = Layer::LogUpGeneric {
            numerators: DenseMultilinearPolynomial::new(numerators),
            denominators: DenseMultilinearPolynomial::new(denominators),
        };
        let artifact = prove_and_verify_logup(input_layer);
        assert_eq!(artifact.n_variables, 3);
    }

    // ---- Negative tests: corrupt proofs should be rejected ----

    #[test]
    fn corrupt_output_claims_rejected() {
        let values: Vec<FE> = (1u64..=4).map(FE::from).collect();
        let input_layer = Layer::GrandProduct(DenseMultilinearPolynomial::new(values));

        let mut prover_transcript = DefaultTranscript::<F>::new(&[]);
        let (mut proof, _) = prove(&mut prover_transcript, input_layer).unwrap();

        // Corrupt the output claims
        proof.output_claims[0] = FE::from(999);

        let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
        let result = verifier::verify(Gate::GrandProduct, &proof, &mut verifier_transcript);
        assert!(result.is_err());
    }

    #[test]
    fn corrupt_sumcheck_proof_rejected() {
        let values: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let input_layer = Layer::GrandProduct(DenseMultilinearPolynomial::new(values));

        let mut prover_transcript = DefaultTranscript::<F>::new(&[]);
        let (mut proof, _) = prove(&mut prover_transcript, input_layer).unwrap();

        // Find a sumcheck proof with non-empty round polys and corrupt it
        let idx = proof
            .sumcheck_proofs
            .iter()
            .position(|p| !p.round_polys.is_empty())
            .expect("should have at least one non-trivial sumcheck");
        proof.sumcheck_proofs[idx].round_polys[0] = Polynomial::new(&[FE::from(1), FE::from(2)]);

        let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
        let result = verifier::verify(Gate::GrandProduct, &proof, &mut verifier_transcript);
        assert!(result.is_err());
    }

    #[test]
    fn corrupt_mask_rejected() {
        let values: Vec<FE> = (1u64..=4).map(FE::from).collect();
        let input_layer = Layer::GrandProduct(DenseMultilinearPolynomial::new(values));

        let mut prover_transcript = DefaultTranscript::<F>::new(&[]);
        let (mut proof, _) = prove(&mut prover_transcript, input_layer).unwrap();

        // Corrupt the first mask
        proof.layer_masks[0] = LayerMask::new(vec![[FE::from(42), FE::from(43)]]);

        let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
        let result = verifier::verify(Gate::GrandProduct, &proof, &mut verifier_transcript);
        assert!(result.is_err());
    }

    #[test]
    fn wrong_gate_type_rejected() {
        // Prove with GrandProduct but verify with LogUp gate
        let values: Vec<FE> = (1u64..=4).map(FE::from).collect();
        let input_layer = Layer::GrandProduct(DenseMultilinearPolynomial::new(values));

        let mut prover_transcript = DefaultTranscript::<F>::new(&[]);
        let (proof, _) = prove(&mut prover_transcript, input_layer).unwrap();

        let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
        let result = verifier::verify(Gate::LogUp, &proof, &mut verifier_transcript);
        assert!(result.is_err());
    }

    // ---- Batch prove + verify tests ----

    /// Helper: batch prove and verify, returning the batch artifact.
    fn prove_and_verify_batch(
        gates: Vec<Gate>,
        input_layers: Vec<Layer<F>>,
    ) -> verifier::BatchVerificationResult<F> {
        let mut prover_transcript = DefaultTranscript::<F>::new(&[]);
        let (proof, _artifact) = prove_batch(&mut prover_transcript, input_layers).unwrap();

        let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
        verifier::verify_batch(&gates, &proof, &mut verifier_transcript)
            .expect("batch verification should succeed")
    }

    #[test]
    fn batch_two_grand_products_same_size() {
        let values0: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let values1: Vec<FE> = (11u64..=18).map(FE::from).collect();

        let artifact = prove_and_verify_batch(
            vec![Gate::GrandProduct, Gate::GrandProduct],
            vec![
                Layer::GrandProduct(DenseMultilinearPolynomial::new(values0)),
                Layer::GrandProduct(DenseMultilinearPolynomial::new(values1)),
            ],
        );

        assert_eq!(artifact.n_variables_by_instance, vec![3, 3]);
        assert_eq!(artifact.claims_to_verify_by_instance.len(), 2);
        assert_eq!(artifact.claims_to_verify_by_instance[0].len(), 1);
        assert_eq!(artifact.claims_to_verify_by_instance[1].len(), 1);
    }

    #[test]
    fn batch_two_grand_products_different_sizes() {
        // Instance 0: 2^5 = 32 elements (5 variables)
        let values0: Vec<FE> = (1u64..=32).map(FE::from).collect();
        // Instance 1: 2^3 = 8 elements (3 variables)
        let values1: Vec<FE> = (1u64..=8).map(FE::from).collect();

        let artifact = prove_and_verify_batch(
            vec![Gate::GrandProduct, Gate::GrandProduct],
            vec![
                Layer::GrandProduct(DenseMultilinearPolynomial::new(values0)),
                Layer::GrandProduct(DenseMultilinearPolynomial::new(values1)),
            ],
        );

        assert_eq!(artifact.n_variables_by_instance, vec![5, 3]);
        assert_eq!(artifact.claims_to_verify_by_instance.len(), 2);
    }

    #[test]
    fn batch_grand_product_and_logup() {
        let gp_values: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let logup_nums: Vec<FE> = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];
        let logup_dens: Vec<FE> = vec![FE::from(5), FE::from(6), FE::from(7), FE::from(8)];

        let artifact = prove_and_verify_batch(
            vec![Gate::GrandProduct, Gate::LogUp],
            vec![
                Layer::GrandProduct(DenseMultilinearPolynomial::new(gp_values)),
                Layer::LogUpGeneric {
                    numerators: DenseMultilinearPolynomial::new(logup_nums),
                    denominators: DenseMultilinearPolynomial::new(logup_dens),
                },
            ],
        );

        assert_eq!(artifact.n_variables_by_instance, vec![3, 2]);
        // GrandProduct has 1 column, LogUp has 2 columns
        assert_eq!(artifact.claims_to_verify_by_instance[0].len(), 1);
        assert_eq!(artifact.claims_to_verify_by_instance[1].len(), 2);
    }

    #[test]
    fn batch_three_instances_mixed() {
        let gp_values: Vec<FE> = (1u64..=16).map(FE::from).collect();
        let logup_nums: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let logup_dens: Vec<FE> = (11u64..=18).map(FE::from).collect();
        let singles_dens: Vec<FE> = vec![FE::from(2), FE::from(3), FE::from(5), FE::from(7)];

        let artifact = prove_and_verify_batch(
            vec![Gate::GrandProduct, Gate::LogUp, Gate::LogUp],
            vec![
                Layer::GrandProduct(DenseMultilinearPolynomial::new(gp_values)),
                Layer::LogUpGeneric {
                    numerators: DenseMultilinearPolynomial::new(logup_nums),
                    denominators: DenseMultilinearPolynomial::new(logup_dens),
                },
                Layer::LogUpSingles {
                    denominators: DenseMultilinearPolynomial::new(singles_dens),
                },
            ],
        );

        assert_eq!(artifact.n_variables_by_instance, vec![4, 3, 2]);
        assert_eq!(artifact.claims_to_verify_by_instance.len(), 3);
    }

    #[test]
    fn batch_single_instance_matches_single_prove() {
        let values: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let input_layer = Layer::GrandProduct(DenseMultilinearPolynomial::new(values.clone()));

        // Single prove + verify
        let mut p_ch = DefaultTranscript::<F>::new(&[]);
        let (single_proof, _) = prove(&mut p_ch, input_layer.clone()).unwrap();
        let mut v_ch = DefaultTranscript::<F>::new(&[]);
        let single_result = verifier::verify(Gate::GrandProduct, &single_proof, &mut v_ch)
            .expect("single verification should succeed");

        // Batch prove + verify with 1 instance
        let batch_result = prove_and_verify_batch(
            vec![Gate::GrandProduct],
            vec![Layer::GrandProduct(DenseMultilinearPolynomial::new(values))],
        );

        assert_eq!(
            single_result.n_variables,
            batch_result.n_variables_by_instance[0]
        );
        assert_eq!(
            single_result.claims_to_verify.len(),
            batch_result.claims_to_verify_by_instance[0].len()
        );
    }

    // ---- Batch negative tests ----

    #[test]
    fn batch_corrupt_output_claims_rejected() {
        let values0: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let values1: Vec<FE> = (11u64..=18).map(FE::from).collect();

        let mut prover_transcript = DefaultTranscript::<F>::new(&[]);
        let (mut proof, _) = prove_batch(
            &mut prover_transcript,
            vec![
                Layer::GrandProduct(DenseMultilinearPolynomial::new(values0)),
                Layer::GrandProduct(DenseMultilinearPolynomial::new(values1)),
            ],
        )
        .unwrap();

        proof.output_claims_by_instance[0][0] = FE::from(999);

        let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
        let result = verifier::verify_batch(
            &[Gate::GrandProduct, Gate::GrandProduct],
            &proof,
            &mut verifier_transcript,
        );
        assert!(result.is_err());
    }

    #[test]
    fn batch_corrupt_sumcheck_proof_rejected() {
        let values0: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let values1: Vec<FE> = (11u64..=18).map(FE::from).collect();

        let mut prover_transcript = DefaultTranscript::<F>::new(&[]);
        let (mut proof, _) = prove_batch(
            &mut prover_transcript,
            vec![
                Layer::GrandProduct(DenseMultilinearPolynomial::new(values0)),
                Layer::GrandProduct(DenseMultilinearPolynomial::new(values1)),
            ],
        )
        .unwrap();

        let idx = proof
            .sumcheck_proofs
            .iter()
            .position(|p| !p.round_polys.is_empty())
            .expect("should have at least one non-trivial sumcheck");
        proof.sumcheck_proofs[idx].round_polys[0] = Polynomial::new(&[FE::from(1), FE::from(2)]);

        let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
        let result = verifier::verify_batch(
            &[Gate::GrandProduct, Gate::GrandProduct],
            &proof,
            &mut verifier_transcript,
        );
        assert!(result.is_err());
    }

    #[test]
    fn batch_corrupt_mask_rejected() {
        let values0: Vec<FE> = (1u64..=4).map(FE::from).collect();
        let values1: Vec<FE> = (11u64..=14).map(FE::from).collect();

        let mut prover_transcript = DefaultTranscript::<F>::new(&[]);
        let (mut proof, _) = prove_batch(
            &mut prover_transcript,
            vec![
                Layer::GrandProduct(DenseMultilinearPolynomial::new(values0)),
                Layer::GrandProduct(DenseMultilinearPolynomial::new(values1)),
            ],
        )
        .unwrap();

        proof.layer_masks_by_instance[0][0] = LayerMask::new(vec![[FE::from(42), FE::from(43)]]);

        let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
        let result = verifier::verify_batch(
            &[Gate::GrandProduct, Gate::GrandProduct],
            &proof,
            &mut verifier_transcript,
        );
        assert!(result.is_err());
    }

    // ---- Read-only memory tests using LogUp-GKR ----

    /// Builds a read-only memory check using batch LogUp-GKR.
    ///
    /// ROM table: [10, 20, 30, 40]
    /// Accesses:  [20, 10, 20, 30, 10, 20, 40, 30]
    ///
    /// The identity to prove is:
    ///   ∑ 1/(z - a_i) = ∑ m_j/(z - t_j)
    ///
    /// where z is a random challenge, a_i are the accesses, t_j are the table entries,
    /// and m_j are the access multiplicities for each table entry.
    #[test]
    fn read_only_memory_valid_accesses() {
        let z = FE::from(100);

        // Table: values [10, 20, 30, 40], multiplicities [2, 3, 2, 1]
        let table_values: Vec<u64> = vec![10, 20, 30, 40];
        let multiplicities: Vec<u64> = vec![2, 3, 2, 1];

        // Accesses: [20, 10, 20, 30, 10, 20, 40, 30]
        let accesses: Vec<u64> = vec![20, 10, 20, 30, 10, 20, 40, 30];

        // Denominators = z - value
        let access_dens: Vec<FE> = accesses.iter().map(|&a| z - FE::from(a)).collect();
        let table_dens: Vec<FE> = table_values.iter().map(|&t| z - FE::from(t)).collect();
        let table_mults: Vec<FE> = multiplicities.iter().map(|&m| FE::from(m)).collect();

        // Access side: LogUpSingles (8 elements, 3 variables)
        // Each access contributes 1/(z - a_i)
        let access_layer = Layer::LogUpSingles {
            denominators: DenseMultilinearPolynomial::new(access_dens),
        };

        // Table side: LogUpMultiplicities (4 elements, 2 variables)
        // Each entry contributes m_j/(z - t_j)
        let table_layer = Layer::LogUpMultiplicities {
            numerators: DenseMultilinearPolynomial::new(table_mults),
            denominators: DenseMultilinearPolynomial::new(table_dens),
        };

        // Batch prove both instances
        let mut prover_transcript = DefaultTranscript::<F>::new(&[]);
        let (proof, _artifact) =
            prove_batch(&mut prover_transcript, vec![access_layer, table_layer]).unwrap();

        // Batch verify
        let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
        let result = verifier::verify_batch(
            &[Gate::LogUp, Gate::LogUp],
            &proof,
            &mut verifier_transcript,
        );
        assert!(result.is_ok(), "batch verification failed");

        // Check that both instances produce the same fraction:
        // access_num/access_den == table_num/table_den
        // i.e. access_num * table_den == table_num * access_den
        let access_output = &proof.output_claims_by_instance[0]; // [num, den]
        let table_output = &proof.output_claims_by_instance[1]; // [num, den]
        let lhs = access_output[0] * table_output[1];
        let rhs = table_output[0] * access_output[1];
        assert_eq!(lhs, rhs, "ROM check: fractions should be equal");
    }

    /// Same ROM table but with an invalid access (value 50 not in table).
    /// The output fractions should NOT match.
    #[test]
    fn read_only_memory_invalid_access_detected() {
        let z = FE::from(100);

        // Table: [10, 20, 30, 40], multiplicities [2, 3, 2, 1]
        let table_values: Vec<u64> = vec![10, 20, 30, 40];
        let multiplicities: Vec<u64> = vec![2, 3, 2, 1];

        // Accesses: one invalid access (50 instead of 40)
        let accesses: Vec<u64> = vec![20, 10, 20, 30, 10, 20, 50, 30];

        let access_dens: Vec<FE> = accesses.iter().map(|&a| z - FE::from(a)).collect();
        let table_dens: Vec<FE> = table_values.iter().map(|&t| z - FE::from(t)).collect();
        let table_mults: Vec<FE> = multiplicities.iter().map(|&m| FE::from(m)).collect();

        let access_layer = Layer::LogUpSingles {
            denominators: DenseMultilinearPolynomial::new(access_dens),
        };
        let table_layer = Layer::LogUpMultiplicities {
            numerators: DenseMultilinearPolynomial::new(table_mults),
            denominators: DenseMultilinearPolynomial::new(table_dens),
        };

        let mut prover_transcript = DefaultTranscript::<F>::new(&[]);
        let (proof, _artifact) =
            prove_batch(&mut prover_transcript, vec![access_layer, table_layer]).unwrap();

        // GKR itself verifies fine (each instance is internally consistent)
        let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
        let result = verifier::verify_batch(
            &[Gate::LogUp, Gate::LogUp],
            &proof,
            &mut verifier_transcript,
        );
        assert!(result.is_ok(), "GKR verification should still pass");

        // But the ROM check fails: fractions don't match
        let access_output = &proof.output_claims_by_instance[0];
        let table_output = &proof.output_claims_by_instance[1];
        let lhs = access_output[0] * table_output[1];
        let rhs = table_output[0] * access_output[1];
        assert_ne!(lhs, rhs, "ROM check should fail for invalid access");
    }

    // ---- Malformed proof tests (should return errors, not panic) ----

    #[test]
    fn verify_malformed_proof_too_few_sumcheck_rounds() {
        // Issue 1: crafted proof where a layer has too few sumcheck rounds
        // should return MalformedProof, not panic on OOD slice.
        let values: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let input_layer = Layer::GrandProduct(DenseMultilinearPolynomial::new(values));

        let mut prover_transcript = DefaultTranscript::<F>::new(&[]);
        let (mut proof, _) = prove(&mut prover_transcript, input_layer).unwrap();

        // Remove round polynomials from the second layer to make sumcheck_ood_point too short.
        if proof.sumcheck_proofs.len() > 1 && !proof.sumcheck_proofs[1].round_polys.is_empty() {
            proof.sumcheck_proofs[1].round_polys.clear();
        }

        let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
        let result = verifier::verify(Gate::GrandProduct, &proof, &mut verifier_transcript);
        assert!(result.is_err(), "should return error, not panic");
    }

    #[test]
    fn batch_single_element_instance_does_not_panic() {
        // Issue 2: size-1 instance (0 variables, 0 layers) should not panic.
        let single = Layer::GrandProduct(DenseMultilinearPolynomial::new(vec![FE::from(42)]));

        let mut prover_transcript = DefaultTranscript::<F>::new(&[]);
        let (proof, artifact) = prove_batch(&mut prover_transcript, vec![single]).unwrap();

        assert_eq!(artifact.n_variables_by_instance, vec![0]);
        assert_eq!(artifact.claims_to_verify_by_instance[0], vec![FE::from(42)]);

        let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
        let result =
            verifier::verify_batch(&[Gate::GrandProduct], &proof, &mut verifier_transcript);
        assert!(
            result.is_ok(),
            "batch verification of size-1 instance should succeed"
        );
    }

    #[test]
    fn batch_mixed_with_zero_layer_instance() {
        // Mix a size-1 instance with a normal one.
        let single = Layer::GrandProduct(DenseMultilinearPolynomial::new(vec![FE::from(7)]));
        let normal: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let normal_layer = Layer::GrandProduct(DenseMultilinearPolynomial::new(normal));

        let artifact = prove_and_verify_batch(
            vec![Gate::GrandProduct, Gate::GrandProduct],
            vec![single, normal_layer],
        );

        assert_eq!(artifact.n_variables_by_instance, vec![0, 3]);
        assert_eq!(artifact.claims_to_verify_by_instance[0], vec![FE::from(7)]);
    }

    #[test]
    fn logup_multiplicities_two_elements_does_not_panic() {
        // 2-element LogUpMultiplicities (1 variable) — the oracle is already constant,
        // so fix_first_variable is never called to convert to LogUpGeneric.
        // try_into_mask must handle LogUpMultiplicities directly.
        let input_layer = Layer::LogUpMultiplicities {
            numerators: DenseMultilinearPolynomial::new(vec![FE::from(3), FE::from(1)]),
            denominators: DenseMultilinearPolynomial::new(vec![FE::from(10), FE::from(20)]),
        };
        let artifact = prove_and_verify_logup(input_layer);
        assert_eq!(artifact.n_variables, 1);
        assert_eq!(artifact.claims_to_verify.len(), 2);
    }

    #[test]
    fn single_element_wrong_gate_rejected() {
        // A size-1 GrandProduct verified with LogUp gate should fail.
        let input = Layer::GrandProduct(DenseMultilinearPolynomial::new(vec![FE::from(42)]));

        let mut prover_transcript = DefaultTranscript::<F>::new(&[]);
        let (proof, _) = prove(&mut prover_transcript, input).unwrap();

        let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
        let result = verifier::verify(Gate::LogUp, &proof, &mut verifier_transcript);
        assert!(
            result.is_err(),
            "wrong gate on 0-layer instance should be rejected"
        );
    }

    #[test]
    fn batch_single_element_wrong_gate_rejected() {
        // A size-1 GrandProduct in batch verified with LogUp gate should fail.
        let single = Layer::GrandProduct(DenseMultilinearPolynomial::new(vec![FE::from(42)]));

        let mut prover_transcript = DefaultTranscript::<F>::new(&[]);
        let (proof, _) = prove_batch(&mut prover_transcript, vec![single]).unwrap();

        let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
        let result = verifier::verify_batch(&[Gate::LogUp], &proof, &mut verifier_transcript);
        assert!(
            result.is_err(),
            "wrong gate on 0-layer batch instance should be rejected"
        );
    }

    // ---- Edge case: empty and minimal inputs ----

    #[test]
    fn batch_empty_instances() {
        // Zero instances: prove_batch returns empty proof, verify_batch accepts it.
        let mut prover_transcript = DefaultTranscript::<F>::new(&[]);
        let (proof, artifact) = prove_batch(&mut prover_transcript, vec![]).unwrap();

        assert!(proof.sumcheck_proofs.is_empty());
        assert!(proof.layer_masks_by_instance.is_empty());
        assert!(proof.output_claims_by_instance.is_empty());
        assert!(artifact.ood_point.is_empty());
        assert!(artifact.claims_to_verify_by_instance.is_empty());

        let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
        let result = verifier::verify_batch(&[], &proof, &mut verifier_transcript);
        assert!(result.is_ok(), "empty batch should verify successfully");

        let vr = result.unwrap();
        assert!(vr.claims_to_verify_by_instance.is_empty());
    }

    #[test]
    fn single_element_prove_verify_roundtrip() {
        // Size-1 input (0 variables, 0 sumcheck layers) for single prove/verify.
        let input = Layer::GrandProduct(DenseMultilinearPolynomial::new(vec![FE::from(42)]));

        let mut prover_transcript = DefaultTranscript::<F>::new(&[]);
        let (proof, artifact) = prove(&mut prover_transcript, input).unwrap();

        assert_eq!(proof.output_claims, vec![FE::from(42)]);
        assert!(proof.sumcheck_proofs.is_empty());
        assert!(proof.layer_masks.is_empty());
        assert_eq!(artifact.n_variables, 0);
        assert!(artifact.ood_point.is_empty());
        assert_eq!(artifact.claims_to_verify, vec![FE::from(42)]);

        let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
        let result = verifier::verify(Gate::GrandProduct, &proof, &mut verifier_transcript);
        assert!(result.is_ok(), "size-1 single prove should verify");

        let vr = result.unwrap();
        assert_eq!(vr.n_variables, 0);
        assert_eq!(vr.claims_to_verify, vec![FE::from(42)]);
    }

    #[test]
    fn batch_multiple_zero_layer_instances() {
        // Multiple size-1 instances batched together (all have 0 sumcheck layers).
        let a = Layer::GrandProduct(DenseMultilinearPolynomial::new(vec![FE::from(7)]));
        let b = Layer::GrandProduct(DenseMultilinearPolynomial::new(vec![FE::from(13)]));
        let c = Layer::LogUpGeneric {
            numerators: DenseMultilinearPolynomial::new(vec![FE::from(3)]),
            denominators: DenseMultilinearPolynomial::new(vec![FE::from(5)]),
        };

        let artifact = prove_and_verify_batch(
            vec![Gate::GrandProduct, Gate::GrandProduct, Gate::LogUp],
            vec![a, b, c],
        );

        assert_eq!(artifact.n_variables_by_instance, vec![0, 0, 0]);
        assert_eq!(artifact.claims_to_verify_by_instance[0], vec![FE::from(7)]);
        assert_eq!(artifact.claims_to_verify_by_instance[1], vec![FE::from(13)]);
        assert_eq!(artifact.claims_to_verify_by_instance[2].len(), 2);
    }
}
