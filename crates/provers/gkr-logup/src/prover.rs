use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{HasDefaultTranscript, IsField};
use lambdaworks_math::traits::ByteConversion;

use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;

use crate::eq_evals::EqEvaluations;
use crate::fraction::{Fraction, Reciprocal};
use crate::layer::{gen_layers, Layer};
use crate::mle::Mle;
use crate::sumcheck::{self, SumcheckOracle};
use crate::utils::{eq, random_linear_combination};
use crate::verifier::{LayerMask, Proof, VerificationResult};
use lambdaworks_math::polynomial::Polynomial;

/// Multivariate polynomial oracle for the GKR protocol.
///
/// Represents `P(x) = eq(x, y) * gate_output(x)` where gate_output depends on the layer type:
/// - GrandProduct: `inp(x, 0) * inp(x, 1)`
/// - LogUp: `(num(x,0)*den(x,1) + num(x,1)*den(x,0)) + lambda * den(x,0)*den(x,1)`
pub struct LayerOracle<F: IsField> {
    pub eq_evals: EqEvaluations<F>,
    pub input_layer: Layer<F>,
    pub eq_fixed_var_correction: FieldElement<F>,
    pub lambda: FieldElement<F>,
}

impl<F: IsField> LayerOracle<F> {
    fn is_constant(&self) -> bool {
        self.n_variables() == 0
    }

    /// Extracts the mask (column values at the two endpoints) from a constant oracle.
    pub fn try_into_mask(self) -> Option<LayerMask<F>> {
        if !self.is_constant() {
            return None;
        }

        let columns = match self.input_layer {
            Layer::GrandProduct(mle) => {
                vec![[mle.at(0), mle.at(1)]]
            }
            Layer::LogUpGeneric {
                numerators,
                denominators,
            } => {
                vec![
                    [numerators.at(0), numerators.at(1)],
                    [denominators.at(0), denominators.at(1)],
                ]
            }
            Layer::LogUpMultiplicities { .. } => {
                // Should never happen: Multiplicities converts to Generic on first fix
                unreachable!("LogUpMultiplicities should have been converted to LogUpGeneric")
            }
            Layer::LogUpSingles { denominators } => {
                vec![
                    [FieldElement::one(), FieldElement::one()],
                    [denominators.at(0), denominators.at(1)],
                ]
            }
        };

        Some(LayerMask::new(columns))
    }
}

impl<F: IsField> SumcheckOracle<F> for LayerOracle<F> {
    fn n_variables(&self) -> usize {
        self.input_layer.n_variables() - 1
    }

    fn sum_as_poly_in_first_variable(
        &self,
        claim: &FieldElement<F>,
    ) -> Polynomial<FieldElement<F>> {
        let n_variables = self.n_variables();
        assert!(n_variables > 0, "Cannot sum a constant oracle");
        let n_terms = 1 << (n_variables - 1);
        let y = self.eq_evals.y();
        let lambda = &self.lambda;

        let (mut eval_at_0, mut eval_at_2) = match &self.input_layer {
            Layer::GrandProduct(col) => eval_grand_product_sum(&self.eq_evals, col, n_terms),
            Layer::LogUpGeneric {
                numerators,
                denominators,
            } => eval_logup_sum(
                &self.eq_evals,
                Some(numerators),
                denominators,
                n_terms,
                lambda,
            ),
            Layer::LogUpMultiplicities {
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
                eval_logup_singles_sum(&self.eq_evals, denominators, n_terms, lambda)
            }
        };

        eval_at_0 = &eval_at_0 * &self.eq_fixed_var_correction;
        eval_at_2 = &eval_at_2 * &self.eq_fixed_var_correction;

        correct_sum_as_poly_in_first_variable(eval_at_0, eval_at_2, claim, y, n_variables)
    }

    fn fix_first_variable(self, challenge: &FieldElement<F>) -> Self {
        if self.is_constant() {
            return self;
        }

        let y = self.eq_evals.y();
        let z0 = &y[y.len() - self.n_variables()];
        let eq_fixed_var_correction = &self.eq_fixed_var_correction
            * &eq(std::slice::from_ref(challenge), std::slice::from_ref(z0));

        Self {
            eq_evals: self.eq_evals,
            eq_fixed_var_correction,
            input_layer: self.input_layer.fix_first_variable(challenge),
            lambda: self.lambda,
        }
    }
}

/// Evaluates `sum_x eq((0^|r|, 0, x), y) * inp(r, t, x, 0) * inp(r, t, x, 1)` at t=0 and t=2.
fn eval_grand_product_sum<F: IsField>(
    eq_evals: &EqEvaluations<F>,
    input_layer: &Mle<F>,
    n_terms: usize,
) -> (FieldElement<F>, FieldElement<F>) {
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
    input_numerators: Option<&Mle<F>>,
    input_denominators: &Mle<F>,
    n_terms: usize,
    lambda: &FieldElement<F>,
) -> (FieldElement<F>, FieldElement<F>) {
    let mut eval_at_0 = FieldElement::<F>::zero();
    let mut eval_at_2 = FieldElement::<F>::zero();

    for i in 0..n_terms {
        let get_num = |idx: usize| -> FieldElement<F> {
            match input_numerators {
                Some(nums) => nums[idx].clone(),
                None => FieldElement::one(),
            }
        };

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

/// Evaluates the LogUp singles sum at t=0 and t=2 (numerators are all 1).
fn eval_logup_singles_sum<F: IsField>(
    eq_evals: &EqEvaluations<F>,
    input_denominators: &Mle<F>,
    n_terms: usize,
    lambda: &FieldElement<F>,
) -> (FieldElement<F>, FieldElement<F>) {
    let mut eval_at_0 = FieldElement::<F>::zero();
    let mut eval_at_2 = FieldElement::<F>::zero();

    for i in 0..n_terms {
        let den_r0i0 = &input_denominators[i * 2];
        let den_r0i1 = &input_denominators[i * 2 + 1];
        let den_r1i0 = &input_denominators[(n_terms + i) * 2];
        let den_r1i1 = &input_denominators[(n_terms + i) * 2 + 1];

        let den_r2i0 = &(den_r1i0 + den_r1i0) - den_r0i0;
        let den_r2i1 = &(den_r1i1 + den_r1i1) - den_r0i1;

        // 1/a + 1/b = (a+b)/(a*b)
        let frac_r0 = Reciprocal::new(den_r0i0.clone()) + Reciprocal::new(den_r0i1.clone());
        let frac_r2 = Reciprocal::new(den_r2i0) + Reciprocal::new(den_r2i1);

        let eq_eval = &eq_evals[i];
        eval_at_0 =
            &eval_at_0 + &(eq_eval * &(&frac_r0.numerator + &(lambda * &frac_r0.denominator)));
        eval_at_2 =
            &eval_at_2 + &(eq_eval * &(&frac_r2.numerator + &(lambda * &frac_r2.denominator)));
    }

    (eval_at_0, eval_at_2)
}

/// Computes `r(t) = sum_x eq((t, x), y[-k:]) * p(t, x)` from evaluations of
/// `f(t) = sum_x eq(({0}^(n - k), 0, x), y) * p(t, x)`.
///
/// Uses 4-point Lagrange interpolation at {0, 1, 2, b} where b is the root
/// of eq(t, y[n-k]).
///
/// See <https://ia.cr/2024/108> (section 3.2).
fn correct_sum_as_poly_in_first_variable<F: IsField>(
    f_at_0: FieldElement<F>,
    f_at_2: FieldElement<F>,
    claim: &FieldElement<F>,
    y: &[FieldElement<F>],
    k: usize,
) -> Polynomial<FieldElement<F>> {
    assert!(k > 0);
    let n = y.len();
    assert!(k <= n);

    // a_const = 1 / eq((0^(n-k+1)), y[..n-k+1])
    let zeros: Vec<FieldElement<F>> = vec![FieldElement::zero(); n - k + 1];
    let a_const = eq(&zeros, &y[..n - k + 1]).inv().unwrap();

    // Find the root of eq(t, y[n-k]):
    //   eq(t, y[n-k]) = t * y[n-k] + (1-t)*(1-y[n-k])
    //   0 = 1 - y[n-k] - t*(1 - 2*y[n-k])
    //   t = (1 - y[n-k]) / (1 - 2*y[n-k])
    let one = FieldElement::<F>::one();
    let two = &one + &one;
    let y_nk = &y[n - k];
    let b_const = (&one - y_nk) * (&one - &(&two * y_nk)).inv().unwrap();

    // r(0) = f(0) * eq(0, y[n-k]) * a_const
    let zero = FieldElement::<F>::zero();
    let eq_at_0 = eq(std::slice::from_ref(&zero), std::slice::from_ref(y_nk));
    let r_at_0 = &f_at_0 * &eq_at_0 * &a_const;

    // r(1) = claim - r(0) (since claim = r(0) + r(1))
    let r_at_1 = claim - &r_at_0;

    // r(2) = f(2) * eq(2, y[n-k]) * a_const
    let eq_at_2 = eq(std::slice::from_ref(&two), std::slice::from_ref(y_nk));
    let r_at_2 = &f_at_2 * &eq_at_2 * &a_const;

    // r(b) = 0 (b is the root of eq(t, y[n-k]))
    let r_at_b = FieldElement::zero();

    // Interpolate degree-3 polynomial through (0, r(0)), (1, r(1)), (2, r(2)), (b, 0)
    Polynomial::interpolate(
        &[FieldElement::zero(), one, two, b_const],
        &[r_at_0, r_at_1, r_at_2, r_at_b],
    )
    .expect("interpolation points are distinct")
}

/// Proves a single GKR instance.
///
/// The input layer should be committed to the channel before calling this function.
///
/// Returns a `Proof` and a `VerificationResult` containing the OOD point and claims
/// to verify against the input layer's MLE.
pub fn prove<F, T>(channel: &mut T, input_layer: Layer<F>) -> (Proof<F>, VerificationResult<F>)
where
    F: IsField + HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
    T: IsTranscript<F>,
{
    // Generate all layers from input (leaves) to output (root)
    let layers = gen_layers(input_layer);
    let n_layers = layers.len() - 1; // number of transitions

    // Get output values from the root layer
    let output_claims = layers
        .last()
        .unwrap()
        .try_into_output_layer_values()
        .expect("last layer should be output");

    // Append output claims to channel
    for claim in &output_claims {
        channel.append_field_element(claim);
    }

    // Sample lambda for combining columns
    let lambda: FieldElement<F> = channel.sample_field_element();

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

        let oracle = LayerOracle {
            eq_evals,
            input_layer: layer.clone(),
            eq_fixed_var_correction: FieldElement::one(),
            lambda: lambda.clone(),
        };

        // Run sumcheck
        let (sumcheck_proof, sumcheck_ood_point, constant_oracle) =
            sumcheck::prove(claim, oracle, channel);

        // Extract mask from the constant oracle
        let mask = constant_oracle
            .try_into_mask()
            .expect("oracle should be constant after sumcheck");

        // Append mask to channel
        for col in mask.columns() {
            channel.append_field_element(&col[0]);
            channel.append_field_element(&col[1]);
        }

        // Sample challenge for next layer
        let challenge: FieldElement<F> = channel.sample_field_element();

        // Update ood_point
        ood_point = sumcheck_ood_point;
        ood_point.push(challenge.clone());

        // Reduce mask at challenge point to get claims for next layer
        claims_to_verify = mask.reduce_at_point(&challenge);

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

    (proof, artifact)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mle::Mle;
    use crate::verifier::{self, Gate, LayerMask};
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    const MODULUS: u64 = 2013265921;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    /// Helper: prove and verify a GrandProduct layer, returning the artifact.
    fn prove_and_verify_grand_product(values: Vec<FE>) -> VerificationResult<F> {
        let input_layer = Layer::GrandProduct(Mle::new(values));

        let mut prover_channel = DefaultTranscript::<F>::new(&[]);
        let (proof, _artifact) = prove(&mut prover_channel, input_layer);

        let mut verifier_channel = DefaultTranscript::<F>::new(&[]);
        verifier::verify(Gate::GrandProduct, &proof, &mut verifier_channel)
            .expect("verification should succeed")
    }

    /// Helper: prove and verify a LogUp layer, returning the artifact.
    fn prove_and_verify_logup(input_layer: Layer<F>) -> VerificationResult<F> {
        let mut prover_channel = DefaultTranscript::<F>::new(&[]);
        let (proof, _artifact) = prove(&mut prover_channel, input_layer);

        let mut verifier_channel = DefaultTranscript::<F>::new(&[]);
        verifier::verify(Gate::LogUp, &proof, &mut verifier_channel)
            .expect("verification should succeed")
    }

    #[test]
    fn grand_product_prove() {
        let values: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let product: FE = values.iter().cloned().reduce(|a, b| a * b).unwrap();

        let input_layer = Layer::GrandProduct(Mle::new(values));
        let mut channel = DefaultTranscript::<F>::new(&[]);
        let (proof, artifact) = prove(&mut channel, input_layer);

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
            .map(|(n, d)| Fraction::new(n.clone(), d.clone()))
            .sum();

        let input_layer = Layer::LogUpGeneric {
            numerators: Mle::new(numerators),
            denominators: Mle::new(denominators),
        };
        let mut channel = DefaultTranscript::<F>::new(&[]);
        let (proof, _artifact) = prove(&mut channel, input_layer);

        assert_eq!(proof.output_claims.len(), 2);
        let out_ratio = &proof.output_claims[0] * expected.denominator.inv().unwrap();
        let exp_ratio = &expected.numerator * proof.output_claims[1].inv().unwrap();
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
            numerators: Mle::new(numerators),
            denominators: Mle::new(denominators),
        };
        let artifact = prove_and_verify_logup(input_layer);
        assert_eq!(artifact.n_variables, 2);
        assert_eq!(artifact.claims_to_verify.len(), 2);
    }

    #[test]
    fn logup_singles_prove_verify_roundtrip() {
        let denominators: Vec<FE> = vec![FE::from(2), FE::from(3), FE::from(5), FE::from(7)];
        let input_layer = Layer::LogUpSingles {
            denominators: Mle::new(denominators),
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
            numerators: Mle::new(numerators),
            denominators: Mle::new(denominators),
        };
        let artifact = prove_and_verify_logup(input_layer);
        assert_eq!(artifact.n_variables, 2);
    }

    #[test]
    fn logup_singles_prove_verify_size_8() {
        let denominators: Vec<FE> = (2u64..=9).map(FE::from).collect();
        let input_layer = Layer::LogUpSingles {
            denominators: Mle::new(denominators),
        };
        let artifact = prove_and_verify_logup(input_layer);
        assert_eq!(artifact.n_variables, 3);
    }

    #[test]
    fn logup_generic_prove_verify_size_8() {
        let numerators: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let denominators: Vec<FE> = (11u64..=18).map(FE::from).collect();
        let input_layer = Layer::LogUpGeneric {
            numerators: Mle::new(numerators),
            denominators: Mle::new(denominators),
        };
        let artifact = prove_and_verify_logup(input_layer);
        assert_eq!(artifact.n_variables, 3);
    }

    // ---- Negative tests: corrupt proofs should be rejected ----

    #[test]
    fn corrupt_output_claims_rejected() {
        let values: Vec<FE> = (1u64..=4).map(FE::from).collect();
        let input_layer = Layer::GrandProduct(Mle::new(values));

        let mut prover_channel = DefaultTranscript::<F>::new(&[]);
        let (mut proof, _) = prove(&mut prover_channel, input_layer);

        // Corrupt the output claims
        proof.output_claims[0] = FE::from(999);

        let mut verifier_channel = DefaultTranscript::<F>::new(&[]);
        let result = verifier::verify(Gate::GrandProduct, &proof, &mut verifier_channel);
        assert!(result.is_err());
    }

    #[test]
    fn corrupt_sumcheck_proof_rejected() {
        let values: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let input_layer = Layer::GrandProduct(Mle::new(values));

        let mut prover_channel = DefaultTranscript::<F>::new(&[]);
        let (mut proof, _) = prove(&mut prover_channel, input_layer);

        // Find a sumcheck proof with non-empty round polys and corrupt it
        let idx = proof
            .sumcheck_proofs
            .iter()
            .position(|p| !p.round_polys.is_empty())
            .expect("should have at least one non-trivial sumcheck");
        proof.sumcheck_proofs[idx].round_polys[0] = Polynomial::new(&[FE::from(1), FE::from(2)]);

        let mut verifier_channel = DefaultTranscript::<F>::new(&[]);
        let result = verifier::verify(Gate::GrandProduct, &proof, &mut verifier_channel);
        assert!(result.is_err());
    }

    #[test]
    fn corrupt_mask_rejected() {
        let values: Vec<FE> = (1u64..=4).map(FE::from).collect();
        let input_layer = Layer::GrandProduct(Mle::new(values));

        let mut prover_channel = DefaultTranscript::<F>::new(&[]);
        let (mut proof, _) = prove(&mut prover_channel, input_layer);

        // Corrupt the first mask
        proof.layer_masks[0] = LayerMask::new(vec![[FE::from(42), FE::from(43)]]);

        let mut verifier_channel = DefaultTranscript::<F>::new(&[]);
        let result = verifier::verify(Gate::GrandProduct, &proof, &mut verifier_channel);
        assert!(result.is_err());
    }

    #[test]
    fn wrong_gate_type_rejected() {
        // Prove with GrandProduct but verify with LogUp gate
        let values: Vec<FE> = (1u64..=4).map(FE::from).collect();
        let input_layer = Layer::GrandProduct(Mle::new(values));

        let mut prover_channel = DefaultTranscript::<F>::new(&[]);
        let (proof, _) = prove(&mut prover_channel, input_layer);

        let mut verifier_channel = DefaultTranscript::<F>::new(&[]);
        let result = verifier::verify(Gate::LogUp, &proof, &mut verifier_channel);
        assert!(result.is_err());
    }
}
