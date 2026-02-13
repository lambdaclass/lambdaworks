use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{HasDefaultTranscript, IsField};
use lambdaworks_math::traits::ByteConversion;

use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;

use crate::fraction::Fraction;
use crate::sumcheck::{self, SumcheckProof};
use crate::utils::{eq, fold_mle_evals, random_linear_combination};

/// Defines how a 2-to-1 gate operates locally on two input rows.
#[derive(Debug, Clone, Copy)]
pub enum Gate {
    LogUp,
    GrandProduct,
}

impl Gate {
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
        let (sumcheck_ood_point, sumcheck_eval) =
            sumcheck::partially_verify(claim, &sumcheck_proofs[layer], channel)
                .map_err(|_| VerifierError::SumcheckFailed { layer })?;

        // Evaluate the gate locally using the mask
        let mask = &layer_masks[layer];
        let gate_output = gate.eval(mask)?;

        // Compute eq(ood_point, sumcheck_ood_point)
        let eq_eval = if ood_point.is_empty() {
            FieldElement::<F>::one()
        } else {
            eq(&ood_point, &sumcheck_ood_point[..ood_point.len()])
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

        // Update ood_point
        ood_point = sumcheck_ood_point;
        ood_point.push(challenge.clone());

        // Reduce mask at challenge point
        claims_to_verify = mask.reduce_at_point(&challenge);
    }

    Ok(VerificationResult {
        ood_point,
        claims_to_verify,
        n_variables: n_layers,
    })
}
