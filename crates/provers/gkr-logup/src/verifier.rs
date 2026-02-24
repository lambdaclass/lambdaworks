use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{HasDefaultTranscript, IsField};
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::ByteConversion;

use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;

use crate::fraction::Fraction;
use crate::MAX_DEGREE;
use lambdaworks_math::polynomial::eq_eval;

use crate::utils::{fold_mle_evals, random_linear_combination};

/// Proof for the sumcheck protocol: one round polynomial per variable.
#[derive(Debug, Clone)]
pub struct SumcheckProof<F: IsField> {
    pub round_polys: Vec<Polynomial<FieldElement<F>>>,
}

/// Defines how a 2-to-1 gate operates locally on two input rows.
#[derive(Debug, Clone, Copy)]
pub enum Gate {
    LogUp,
    GrandProduct,
}

impl Gate {
    /// Number of columns expected for this gate type.
    pub fn n_columns(&self) -> usize {
        match self {
            Self::GrandProduct => 1,
            Self::LogUp => 2,
        }
    }

    /// Evaluates the gate on the given mask values, returning the output column values.
    pub fn eval<F: IsField>(
        &self,
        mask: &LayerMask<F>,
    ) -> Result<Vec<FieldElement<F>>, VerifierError<F>> {
        match self {
            Self::GrandProduct => {
                if mask.columns().len() != 1 {
                    return Err(VerifierError::InvalidMask);
                }
                let [a, b] = &mask.columns()[0];
                Ok(vec![a * b])
            }
            Self::LogUp => {
                if mask.columns().len() != 2 {
                    return Err(VerifierError::InvalidMask);
                }
                let [num_a, num_b] = &mask.columns()[0];
                let [den_a, den_b] = &mask.columns()[1];
                let a = Fraction::new(num_a.clone(), den_a.clone());
                let b = Fraction::new(num_b.clone(), den_b.clone());
                let res = a + b;
                Ok(vec![res.numerator, res.denominator])
            }
        }
    }
}

/// Stores two evaluations (at 0 and 1) of each column in a GKR layer.
#[derive(Debug, Clone)]
pub struct LayerMask<F: IsField> {
    columns: Vec<[FieldElement<F>; 2]>,
}

impl<F: IsField> LayerMask<F> {
    pub fn new(columns: Vec<[FieldElement<F>; 2]>) -> Self {
        Self { columns }
    }

    pub fn columns(&self) -> &[[FieldElement<F>; 2]] {
        &self.columns
    }

    /// Reduces each column at the challenge point: `fold_mle_evals(x, v0, v1)`.
    pub fn reduce_at_point(&self, x: &FieldElement<F>) -> Vec<FieldElement<F>> {
        self.columns
            .iter()
            .map(|[v0, v1]| fold_mle_evals(x, v0, v1))
            .collect()
    }
}

/// Single-instance GKR proof.
#[derive(Debug, Clone)]
pub struct Proof<F: IsField> {
    /// One sumcheck proof per layer (output to input).
    pub sumcheck_proofs: Vec<SumcheckProof<F>>,
    /// Mask for each layer.
    pub layer_masks: Vec<LayerMask<F>>,
    /// Column values at the output (root) layer.
    pub output_claims: Vec<FieldElement<F>>,
}

/// Values of interest obtained from the GKR protocol execution.
#[derive(Debug, Clone)]
pub struct VerificationResult<F: IsField> {
    /// Out-of-domain point for evaluating columns in the input layer.
    pub ood_point: Vec<FieldElement<F>>,
    /// The claimed evaluations at `ood_point` for each column in the input layer.
    pub claims_to_verify: Vec<FieldElement<F>>,
    /// Number of variables that interpolate the input layer.
    pub n_variables: usize,
}

/// Error encountered during GKR protocol verification.
#[derive(Debug)]
pub enum VerifierError<F: IsField> {
    MalformedProof,
    InvalidMask,
    SumcheckFailed {
        layer: usize,
    },
    CircuitCheckFailure {
        claim: FieldElement<F>,
        output: FieldElement<F>,
        layer: usize,
    },
}

impl<F: IsField> core::fmt::Display for VerifierError<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MalformedProof => write!(f, "proof data is invalid"),
            Self::InvalidMask => write!(f, "mask has wrong number of columns"),
            Self::SumcheckFailed { layer } => {
                write!(f, "sumcheck failed in layer {layer}")
            }
            Self::CircuitCheckFailure { layer, .. } => {
                write!(f, "circuit check failed in layer {layer}")
            }
        }
    }
}

/// Partially verifies a single-instance GKR proof.
///
/// On success returns a `VerificationResult` with the OOD point and claimed evaluations
/// at the input layer. These claims must be verified externally against the actual
/// input layer MLE evaluations.
pub fn verify<F, T>(
    gate: Gate,
    proof: &Proof<F>,
    channel: &mut T,
) -> Result<VerificationResult<F>, VerifierError<F>>
where
    F: IsField + HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
    T: IsTranscript<F>,
{
    let Proof {
        sumcheck_proofs,
        layer_masks,
        output_claims,
    } = proof;

    let n_layers = sumcheck_proofs.len();
    if layer_masks.len() != n_layers {
        return Err(VerifierError::MalformedProof);
    }

    // Validate output claims match gate arity.
    if output_claims.len() != gate.n_columns() {
        return Err(VerifierError::MalformedProof);
    }

    // Append output claims to channel (same as prover)
    for claim in output_claims {
        channel.append_field_element(claim);
    }

    // Sample lambda (same as prover)
    let lambda: FieldElement<F> = channel.sample_field_element();

    let mut ood_point: Vec<FieldElement<F>> = Vec::new();
    let mut claims_to_verify = output_claims.clone();

    for layer in 0..n_layers {
        // Compute claim as random linear combination
        let claim = random_linear_combination(&claims_to_verify, &lambda);

        // Partially verify sumcheck
        let (sumcheck_ood_point, sumcheck_eval) = lambdaworks_sumcheck::partially_verify(
            claim,
            &sumcheck_proofs[layer].round_polys,
            MAX_DEGREE,
            channel,
        )
        .map_err(|_| VerifierError::SumcheckFailed { layer })?;

        // Evaluate the gate locally using the mask
        let mask = &layer_masks[layer];
        let gate_output = gate.eval(mask)?;

        // Compute eq_eval(ood_point, sumcheck_ood_point).
        // On the first layer there is no prior out-of-domain point, so the
        // equality polynomial evaluates to 1 (empty product).
        let eq_eval = if ood_point.is_empty() {
            FieldElement::<F>::one()
        } else if sumcheck_ood_point.len() < ood_point.len() {
            return Err(VerifierError::MalformedProof);
        } else {
            eq_eval(&ood_point, &sumcheck_ood_point[..ood_point.len()])
        };

        // Expected evaluation: eq_eval * random_linear_combination(gate_output, lambda)
        let layer_eval = &eq_eval * &random_linear_combination(&gate_output, &lambda);

        if sumcheck_eval != layer_eval {
            return Err(VerifierError::CircuitCheckFailure {
                claim: sumcheck_eval,
                output: layer_eval,
                layer,
            });
        }

        // Append mask to channel (same as prover)
        for col in mask.columns() {
            channel.append_field_element(&col[0]);
            channel.append_field_element(&col[1]);
        }

        // Sample challenge (same as prover)
        let challenge: FieldElement<F> = channel.sample_field_element();

        // Reduce mask at challenge point (borrow challenge before moving it)
        claims_to_verify = mask.reduce_at_point(&challenge);

        // Update ood_point
        ood_point = sumcheck_ood_point;
        ood_point.push(challenge);
    }

    Ok(VerificationResult {
        ood_point,
        claims_to_verify,
        n_variables: n_layers,
    })
}

/// Batch GKR proof for multiple instances proved simultaneously.
#[derive(Debug, Clone)]
pub struct BatchProof<F: IsField> {
    /// One sumcheck proof per layer (shared across instances).
    pub sumcheck_proofs: Vec<SumcheckProof<F>>,
    /// Mask for each layer for each instance.
    pub layer_masks_by_instance: Vec<Vec<LayerMask<F>>>,
    /// Column circuit outputs for each instance.
    pub output_claims_by_instance: Vec<Vec<FieldElement<F>>>,
}

/// Values of interest obtained from batch GKR protocol execution.
#[derive(Debug, Clone)]
pub struct BatchVerificationResult<F: IsField> {
    /// Shared out-of-domain point for evaluating columns in input layers.
    pub ood_point: Vec<FieldElement<F>>,
    /// The claimed evaluations at `ood_point` for each column in each instance's input layer.
    pub claims_to_verify_by_instance: Vec<Vec<FieldElement<F>>>,
    /// Number of variables in each instance's input layer.
    pub n_variables_by_instance: Vec<usize>,
}

/// Partially verifies a batch GKR proof.
///
/// On success returns a `BatchVerificationResult` with the shared OOD point and per-instance
/// claimed evaluations. These claims must be verified externally against the actual input
/// layer MLE evaluations.
pub fn verify_batch<F, T>(
    gates: &[Gate],
    proof: &BatchProof<F>,
    channel: &mut T,
) -> Result<BatchVerificationResult<F>, VerifierError<F>>
where
    F: IsField + HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
    T: IsTranscript<F>,
{
    let BatchProof {
        sumcheck_proofs,
        layer_masks_by_instance,
        output_claims_by_instance,
    } = proof;

    if layer_masks_by_instance.len() != output_claims_by_instance.len() {
        return Err(VerifierError::MalformedProof);
    }

    let n_instances = layer_masks_by_instance.len();
    if gates.len() != n_instances {
        return Err(VerifierError::MalformedProof);
    }

    // Domain separation: must match prover (see prove_batch).
    channel.append_bytes(b"gkr_batch");
    channel.append_bytes(&(n_instances as u64).to_le_bytes());

    let instance_n_layers = |instance: usize| layer_masks_by_instance[instance].len();
    let n_layers = (0..n_instances).map(instance_n_layers).max().unwrap_or(0);

    if n_layers != sumcheck_proofs.len() {
        return Err(VerifierError::MalformedProof);
    }

    // Validate output claims match gate arity for each instance.
    for (instance, claims) in output_claims_by_instance.iter().enumerate() {
        if claims.len() != gates[instance].n_columns() {
            return Err(VerifierError::MalformedProof);
        }
    }

    let mut ood_point: Vec<FieldElement<F>> = Vec::new();
    let mut claims_to_verify_by_instance: Vec<Option<Vec<FieldElement<F>>>> =
        vec![None; n_instances];

    // Handle zero-layer instances (size-1 inputs: no sumcheck layers).
    for instance in 0..n_instances {
        if instance_n_layers(instance) == 0 {
            claims_to_verify_by_instance[instance] =
                Some(output_claims_by_instance[instance].clone());
        }
    }

    for (layer, sumcheck_proof) in sumcheck_proofs.iter().enumerate() {
        let n_remaining_layers = n_layers - layer;

        // Detect output layers.
        for instance in 0..n_instances {
            if instance_n_layers(instance) == n_remaining_layers {
                claims_to_verify_by_instance[instance] =
                    Some(output_claims_by_instance[instance].clone());
            }
        }

        // Seed channel with active claims.
        for claims in claims_to_verify_by_instance.iter().flatten() {
            for claim in claims {
                channel.append_field_element(claim);
            }
        }

        // Sample randomness (must match prover).
        let sumcheck_alpha: FieldElement<F> = channel.sample_field_element();
        let lambda: FieldElement<F> = channel.sample_field_element();

        let mut sumcheck_claims = Vec::new();
        let mut sumcheck_instances = Vec::new();

        // Compute per-instance claims with doubling factor (skip zero-layer instances).
        // Instances with fewer variables are conceptually padded to max_vars by
        // ignoring the first n_unused variables. This multiplies the sum over the
        // Boolean hypercube by 2^n_unused (see prove_batch_sumcheck for details).
        for (instance, claims) in claims_to_verify_by_instance.iter().enumerate() {
            if let Some(claims) = claims {
                if instance_n_layers(instance) == 0 {
                    continue;
                }
                let n_unused = n_layers - instance_n_layers(instance);
                let doubling_factor = FieldElement::<F>::from(1u64 << n_unused);
                let claim = &random_linear_combination(claims, &lambda) * &doubling_factor;
                sumcheck_claims.push(claim);
                sumcheck_instances.push(instance);
            }
        }

        // Verify sumcheck with combined claim.
        let combined_claim = random_linear_combination(&sumcheck_claims, &sumcheck_alpha);
        let (sumcheck_ood_point, sumcheck_eval) = lambdaworks_sumcheck::partially_verify(
            combined_claim,
            &sumcheck_proof.round_polys,
            MAX_DEGREE,
            channel,
        )
        .map_err(|_| VerifierError::SumcheckFailed { layer })?;

        // Evaluate gates locally at sumcheck OOD point.
        let mut layer_evals = Vec::new();
        for &instance in &sumcheck_instances {
            let n_unused = n_layers - instance_n_layers(instance);
            let mask = &layer_masks_by_instance[instance][layer - n_unused];
            let gate_output = gates[instance].eval(mask)?;

            // eq evaluation uses the relevant suffix of the OOD point.
            // When ood_point.len() <= n_unused the instance hasn't accumulated
            // any OOD coordinates yet, so eq evaluates to 1 (empty product).
            let eq_eval = if ood_point.len() <= n_unused {
                FieldElement::<F>::one()
            } else if sumcheck_ood_point.len() < ood_point.len() {
                return Err(VerifierError::MalformedProof);
            } else {
                eq_eval(
                    &ood_point[n_unused..],
                    &sumcheck_ood_point[n_unused..ood_point.len()],
                )
            };

            layer_evals.push(&eq_eval * &random_linear_combination(&gate_output, &lambda));
        }

        let layer_eval = random_linear_combination(&layer_evals, &sumcheck_alpha);

        if sumcheck_eval != layer_eval {
            return Err(VerifierError::CircuitCheckFailure {
                claim: sumcheck_eval,
                output: layer_eval,
                layer,
            });
        }

        // Seed channel with masks (same order as prover).
        for &instance in &sumcheck_instances {
            let n_unused = n_layers - instance_n_layers(instance);
            let mask = &layer_masks_by_instance[instance][layer - n_unused];
            for col in mask.columns() {
                channel.append_field_element(&col[0]);
                channel.append_field_element(&col[1]);
            }
        }

        // Sample challenge, reduce masks (borrow challenge before moving it).
        let challenge: FieldElement<F> = channel.sample_field_element();
        for instance in sumcheck_instances {
            let n_unused = n_layers - instance_n_layers(instance);
            let mask = &layer_masks_by_instance[instance][layer - n_unused];
            claims_to_verify_by_instance[instance] = Some(mask.reduce_at_point(&challenge));
        }

        // Update OOD point.
        ood_point = sumcheck_ood_point;
        ood_point.push(challenge);
    }

    let claims_to_verify_by_instance: Vec<Vec<FieldElement<F>>> = claims_to_verify_by_instance
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(VerifierError::MalformedProof)?;

    Ok(BatchVerificationResult {
        ood_point,
        claims_to_verify_by_instance,
        n_variables_by_instance: (0..n_instances).map(instance_n_layers).collect(),
    })
}
