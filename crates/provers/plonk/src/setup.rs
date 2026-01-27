use std::collections::HashMap;

use crate::constraint_system::{get_permutation, ConstraintSystem, Variable};
use crate::prover::ProverError;
use crate::test_utils::utils::{generate_domain, generate_permutation_coefficients};
use lambdaworks_crypto::commitments::traits::IsCommitmentScheme;
use lambdaworks_crypto::fiat_shamir::{
    default_transcript::DefaultTranscript, is_transcript::IsTranscript,
};
use lambdaworks_math::field::traits::{HasDefaultTranscript, IsFFTField};
use lambdaworks_math::field::{element::FieldElement, traits::IsField};
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::{AsBytes, ByteConversion};

/// Validates that k1 is a valid coset generator for PLONK.
///
/// For the copy constraint permutation to work correctly, the cosets H, k1*H, and k2*H
/// (where k2 = k1^2) must be disjoint. This requires:
/// 1. k1 is not in the domain H (k1^n != 1)
/// 2. k2 is not in the domain H (k2^n != 1)
/// 3. k1 != 1 (trivial case)
///
/// # Arguments
/// * `k1` - The coset generator
/// * `n` - The size of the domain H
///
/// # Returns
/// * `Ok(())` if k1 is valid
/// * `Err(ProverError::SetupError)` if validation fails
pub fn validate_coset_generator<F: IsField>(
    k1: &FieldElement<F>,
    n: usize,
) -> Result<(), ProverError> {
    // k1 must not be 1 (trivial case)
    if *k1 == FieldElement::one() {
        return Err(ProverError::SetupError(
            "Coset generator k1 cannot be 1".to_string(),
        ));
    }

    // k1 must not be in the domain H (i.e., k1^n != 1)
    let k1_pow_n = k1.pow(n as u64);
    if k1_pow_n == FieldElement::one() {
        return Err(ProverError::SetupError(format!(
            "Coset generator k1 is in the domain H (k1^{} = 1)",
            n
        )));
    }

    // k2 = k1^2 must not be in the domain H (i.e., k2^n != 1)
    let k2 = k1 * k1;
    let k2_pow_n = k2.pow(n as u64);
    if k2_pow_n == FieldElement::one() {
        return Err(ProverError::SetupError(format!(
            "Coset generator k2 = k1^2 is in the domain H (k2^{} = 1)",
            n
        )));
    }

    Ok(())
}

// TODO: implement getters
pub struct Witness<F: IsField> {
    pub a: Vec<FieldElement<F>>,
    pub b: Vec<FieldElement<F>>,
    pub c: Vec<FieldElement<F>>,
}

impl<F: IsField> Witness<F> {
    pub fn new(values: HashMap<Variable, FieldElement<F>>, system: &ConstraintSystem<F>) -> Self {
        let (lro, _) = system.to_matrices();
        let abc: Vec<_> = lro.iter().map(|v| values[v].clone()).collect();
        let n = lro.len() / 3;

        Self {
            a: abc[..n].to_vec(),
            b: abc[n..2 * n].to_vec(),
            c: abc[2 * n..].to_vec(),
        }
    }
}

/// Error types for witness building.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WitnessBuilderError {
    /// A required variable was not assigned a value
    MissingVariable(Variable),
    /// The constraint system reference is missing
    MissingConstraintSystem,
    /// Duplicate assignment to a variable
    DuplicateAssignment(Variable),
}

impl std::fmt::Display for WitnessBuilderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WitnessBuilderError::MissingVariable(v) => {
                write!(f, "Missing value for variable {}", v)
            }
            WitnessBuilderError::MissingConstraintSystem => {
                write!(f, "Constraint system not provided")
            }
            WitnessBuilderError::DuplicateAssignment(v) => {
                write!(f, "Duplicate assignment to variable {}", v)
            }
        }
    }
}

impl std::error::Error for WitnessBuilderError {}

/// Builder for constructing witnesses with validation.
///
/// `WitnessBuilder` provides a safe, ergonomic API for creating PLONK witnesses.
/// It validates that all required variables have values before building.
///
/// # Example
///
/// ```ignore
/// use lambdaworks_plonk::setup::WitnessBuilder;
///
/// let witness = WitnessBuilder::new()
///     .set(x_var, FieldElement::from(4))
///     .set(y_var, FieldElement::from(12))
///     .set(e_var, FieldElement::from(3))
///     .build(&constraint_system)?;
/// ```
pub struct WitnessBuilder<F: IsField> {
    assignments: HashMap<Variable, FieldElement<F>>,
    allow_overwrite: bool,
}

impl<F: IsField> Default for WitnessBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: IsField> WitnessBuilder<F> {
    /// Creates a new empty WitnessBuilder.
    pub fn new() -> Self {
        Self {
            assignments: HashMap::new(),
            allow_overwrite: false,
        }
    }

    /// Creates a new WitnessBuilder from existing assignments.
    pub fn from_assignments(assignments: HashMap<Variable, FieldElement<F>>) -> Self {
        Self {
            assignments,
            allow_overwrite: false,
        }
    }

    /// Allow overwriting existing variable assignments.
    /// By default, attempting to assign to an already-assigned variable returns an error.
    pub fn allow_overwrite(mut self, allow: bool) -> Self {
        self.allow_overwrite = allow;
        self
    }

    /// Assigns a value to a variable.
    ///
    /// Returns an error if the variable already has a value (unless `allow_overwrite` is set).
    pub fn set(
        mut self,
        variable: Variable,
        value: FieldElement<F>,
    ) -> Result<Self, WitnessBuilderError> {
        if !self.allow_overwrite && self.assignments.contains_key(&variable) {
            return Err(WitnessBuilderError::DuplicateAssignment(variable));
        }
        self.assignments.insert(variable, value);
        Ok(self)
    }

    /// Assigns a value to a variable (infallible version for chaining).
    ///
    /// Panics if the variable already has a value (unless `allow_overwrite` is set).
    /// Use `set()` if you want error handling instead of panics.
    pub fn assign(self, variable: Variable, value: FieldElement<F>) -> Self {
        self.set(variable, value)
            .expect("Duplicate variable assignment")
    }

    /// Assigns values to multiple variables from an iterator.
    pub fn set_many(
        mut self,
        assignments: impl IntoIterator<Item = (Variable, FieldElement<F>)>,
    ) -> Result<Self, WitnessBuilderError> {
        for (var, val) in assignments {
            self = self.set(var, val)?;
        }
        Ok(self)
    }

    /// Returns the current assignments.
    pub fn assignments(&self) -> &HashMap<Variable, FieldElement<F>> {
        &self.assignments
    }

    /// Returns a mutable reference to the current assignments.
    pub fn assignments_mut(&mut self) -> &mut HashMap<Variable, FieldElement<F>> {
        &mut self.assignments
    }

    /// Builds the witness from the constraint system.
    ///
    /// This method validates that all variables referenced in the constraint system
    /// have assigned values, then constructs the witness.
    ///
    /// # Errors
    ///
    /// Returns `WitnessBuilderError::MissingVariable` if any required variable
    /// is missing a value.
    pub fn build(self, system: &ConstraintSystem<F>) -> Result<Witness<F>, WitnessBuilderError> {
        let (lro, _) = system.to_matrices();

        // Validate all required variables have values
        for var in &lro {
            if !self.assignments.contains_key(var) {
                return Err(WitnessBuilderError::MissingVariable(*var));
            }
        }

        let abc: Vec<_> = lro
            .iter()
            .map(|v| self.assignments[v].clone())
            .collect();
        let n = lro.len() / 3;

        Ok(Witness {
            a: abc[..n].to_vec(),
            b: abc[n..2 * n].to_vec(),
            c: abc[2 * n..].to_vec(),
        })
    }

    /// Builds the witness using the solver to fill in missing values.
    ///
    /// This method first attempts to solve for any missing variable values using
    /// the constraint system's solver, then builds the witness.
    ///
    /// # Errors
    ///
    /// Returns an error if the solver cannot determine values for all variables.
    pub fn build_with_solver(
        mut self,
        system: &ConstraintSystem<F>,
    ) -> Result<Witness<F>, WitnessBuilderError> {
        // Use the solver to fill in missing values
        match system.solve(self.assignments.clone()) {
            Ok(solved) => {
                self.assignments = solved;
                self.build(system)
            }
            Err(_) => {
                // Try to build anyway and let it fail with specific missing variable
                self.build(system)
            }
        }
    }
}

// TODO: implement getters
#[derive(Clone)]
pub struct CommonPreprocessedInput<F: IsField> {
    pub n: usize,
    /// Number of constraints
    pub domain: Vec<FieldElement<F>>,
    pub omega: FieldElement<F>,
    pub k1: FieldElement<F>,

    pub ql: Polynomial<FieldElement<F>>,
    pub qr: Polynomial<FieldElement<F>>,
    pub qo: Polynomial<FieldElement<F>>,
    pub qm: Polynomial<FieldElement<F>>,
    pub qc: Polynomial<FieldElement<F>>,

    pub s1: Polynomial<FieldElement<F>>,
    pub s2: Polynomial<FieldElement<F>>,
    pub s3: Polynomial<FieldElement<F>>,

    pub s1_lagrange: Vec<FieldElement<F>>,
    pub s2_lagrange: Vec<FieldElement<F>>,
    pub s3_lagrange: Vec<FieldElement<F>>,
}

impl<F: IsFFTField> CommonPreprocessedInput<F> {
    pub fn from_constraint_system(
        system: &ConstraintSystem<F>,
        order_r_minus_1_root_unity: &FieldElement<F>,
    ) -> Result<Self, ProverError> {
        let (lro, q) = system.to_matrices();
        let n = lro.len() / 3;
        let omega = F::get_primitive_root_of_unity(n.trailing_zeros() as u64)
            .map_err(|_| ProverError::PrimitiveRootNotFound(n.trailing_zeros() as u64))?;
        let domain = generate_domain(&omega, n);

        // Validate coset generator k1 ensures disjoint cosets H, k1*H, k2*H
        validate_coset_generator(order_r_minus_1_root_unity, n)?;

        let m = q.len() / 5;
        let ql: Vec<_> = q[..m].to_vec();
        let qr: Vec<_> = q[m..2 * m].to_vec();
        let qm: Vec<_> = q[2 * m..3 * m].to_vec();
        let qo: Vec<_> = q[3 * m..4 * m].to_vec();
        let qc: Vec<_> = q[4 * m..].to_vec();

        let permutation = get_permutation(&lro);
        let permuted =
            generate_permutation_coefficients(&omega, n, &permutation, order_r_minus_1_root_unity);

        let s1_lagrange: Vec<_> = permuted[..n].to_vec();
        let s2_lagrange: Vec<_> = permuted[n..2 * n].to_vec();
        let s3_lagrange: Vec<_> = permuted[2 * n..].to_vec();

        Ok(Self {
            domain,
            n,
            omega,
            k1: order_r_minus_1_root_unity.clone(),
            ql: Polynomial::interpolate_fft::<F>(&ql)
                .map_err(|e| ProverError::FFTError(format!("{:?}", e)))?,
            qr: Polynomial::interpolate_fft::<F>(&qr)
                .map_err(|e| ProverError::FFTError(format!("{:?}", e)))?,
            qo: Polynomial::interpolate_fft::<F>(&qo)
                .map_err(|e| ProverError::FFTError(format!("{:?}", e)))?,
            qm: Polynomial::interpolate_fft::<F>(&qm)
                .map_err(|e| ProverError::FFTError(format!("{:?}", e)))?,
            qc: Polynomial::interpolate_fft::<F>(&qc)
                .map_err(|e| ProverError::FFTError(format!("{:?}", e)))?,
            s1: Polynomial::interpolate_fft::<F>(&s1_lagrange)
                .map_err(|e| ProverError::FFTError(format!("{:?}", e)))?,
            s2: Polynomial::interpolate_fft::<F>(&s2_lagrange)
                .map_err(|e| ProverError::FFTError(format!("{:?}", e)))?,
            s3: Polynomial::interpolate_fft::<F>(&s3_lagrange)
                .map_err(|e| ProverError::FFTError(format!("{:?}", e)))?,
            s1_lagrange,
            s2_lagrange,
            s3_lagrange,
        })
    }
}

pub struct VerificationKey<G1Point> {
    pub qm_1: G1Point,
    pub ql_1: G1Point,
    pub qr_1: G1Point,
    pub qo_1: G1Point,
    pub qc_1: G1Point,

    pub s1_1: G1Point,
    pub s2_1: G1Point,
    pub s3_1: G1Point,
}

pub fn setup<F: IsField, CS: IsCommitmentScheme<F>>(
    common_input: &CommonPreprocessedInput<F>,
    commitment_scheme: &CS,
) -> VerificationKey<CS::Commitment> {
    VerificationKey {
        qm_1: commitment_scheme.commit(&common_input.qm),
        ql_1: commitment_scheme.commit(&common_input.ql),
        qr_1: commitment_scheme.commit(&common_input.qr),
        qo_1: commitment_scheme.commit(&common_input.qo),
        qc_1: commitment_scheme.commit(&common_input.qc),

        s1_1: commitment_scheme.commit(&common_input.s1),
        s2_1: commitment_scheme.commit(&common_input.s2),
        s3_1: commitment_scheme.commit(&common_input.s3),
    }
}

pub fn new_strong_fiat_shamir_transcript<F, CS>(
    vk: &VerificationKey<CS::Commitment>,
    public_input: &[FieldElement<F>],
) -> DefaultTranscript<F>
where
    F: HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
    CS: IsCommitmentScheme<F>,
    CS::Commitment: AsBytes,
{
    let mut transcript = DefaultTranscript::default();

    transcript.append_bytes(&vk.s1_1.as_bytes());
    transcript.append_bytes(&vk.s2_1.as_bytes());
    transcript.append_bytes(&vk.s3_1.as_bytes());
    transcript.append_bytes(&vk.ql_1.as_bytes());
    transcript.append_bytes(&vk.qr_1.as_bytes());
    transcript.append_bytes(&vk.qm_1.as_bytes());
    transcript.append_bytes(&vk.qo_1.as_bytes());
    transcript.append_bytes(&vk.qc_1.as_bytes());

    for value in public_input.iter() {
        transcript.append_field_element(value);
    }

    transcript
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrField;
    use lambdaworks_math::elliptic_curve::{
        short_weierstrass::curves::bls12_381::curve::BLS12381Curve, traits::IsEllipticCurve,
    };
    use lambdaworks_math::field::element::FieldElement as FE;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    use super::*;
    use crate::test_utils::circuit_1::test_common_preprocessed_input_1;
    use crate::test_utils::utils::{test_srs, FpElement, KZG, ORDER_R_MINUS_1_ROOT_UNITY};

    #[test]
    fn test_validate_coset_generator_rejects_one() {
        use lambdaworks_math::field::element::FieldElement;
        let k1 = FieldElement::<FrField>::one();
        let result = validate_coset_generator(&k1, 4);
        assert!(result.is_err());
        assert!(matches!(result, Err(ProverError::SetupError(_))));
    }

    #[test]
    fn test_validate_coset_generator_accepts_valid() {
        use crate::test_utils::utils::ORDER_R_MINUS_1_ROOT_UNITY;
        // The standard test k1 should be valid for n=4
        let result = validate_coset_generator(&ORDER_R_MINUS_1_ROOT_UNITY, 4);
        assert!(result.is_ok());
    }

    #[test]
    fn setup_works_for_simple_circuit() {
        let common_input = test_common_preprocessed_input_1();
        let srs = test_srs(common_input.n);
        let kzg = KZG::new(srs);

        let vk = setup::<FrField, KZG>(&common_input, &kzg);

        let expected_ql = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex_unchecked("1492341357755e31a6306abf3237f84f707ded7cb526b8ffd40901746234ef27f12bc91ef638e4977563db208b765f12"),
            FpElement::from_hex_unchecked("ec3ff8288ea339010658334f494a614f7470c19a08d53a9cf5718e0613bb65d2cdbc1df374057d9b45c35cf1f1b5b72"),
        ).unwrap();
        let expected_qr = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex_unchecked("107ab09b6b8c6fc55087aeb8045e17a6d016bdacbc64476264328e71f3e85a4eacaee34ee963e9c9249b6b1bc9653674"),
            FpElement::from_hex_unchecked("f98e3fe5a53545b67a51da7e7a6cedc51af467abdefd644113fb97edf339aeaa5e2f6a5713725ec76754510b76a10be"),
        ).unwrap();
        let expected_qo = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex_unchecked("9fd00baa112a0064ce5c3c2d243e657b25df8a2f237b91eec27e83157f6ca896a2401d07ec7d7d097d2f2a344e2018f"),
            FpElement::from_hex_unchecked("15922cfa65972d80823c6bb9aeb0637c864b636267bfee2818413e9cdc5f7948575c4ce097bb8b9db8087c4ed5056592"),
        ).unwrap();
        let expected_qm = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex_unchecked("9fd00baa112a0064ce5c3c2d243e657b25df8a2f237b91eec27e83157f6ca896a2401d07ec7d7d097d2f2a344e2018f"),
            FpElement::from_hex_unchecked("46ee4efd3e8b919c8df3bfc949b495ade2be8228bc524974eef94041a517cdbc74fb31e1998746201f683b12afa4519"),
        ).unwrap();

        let expected_s1 = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex_unchecked("187ee12de08728650d18912aa9fe54863922a9eeb37e34ff43041f1d039f00028ad2cdf23705e6f6ab7ea9406535c1b0"),
            FpElement::from_hex_unchecked("4f29051990de0d12b38493992845d9abcb48ef18239eca8b8228618c78ec371d39917bc0d45cf6dc4f79bd64baa9ae2")
        ).unwrap();
        let expected_s2 = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex_unchecked("167c0384025887c01ea704234e813842a4acef7d765c3a94a5442ca685b4fc1d1b425ba7786a7413bd4a7d6a1eb5a35a"),
            FpElement::from_hex_unchecked("12b644100c5d00af27c121806c4779f88e840ff3fdac44124b8175a303d586c4d910486f909b37dda1505c485f053da1")
        ).unwrap();
        let expected_s3 = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex_unchecked("188fb6dba3cf5af8a7f6a44d935bb3dd2083a5beb4c020f581739ebc40659c824a4ca8279cf7d852decfbca572e4fa0e"),
            FpElement::from_hex_unchecked("d84d52582fd95bfa7672f7cef9dd4d0b1b4a54d33f244fdb97df71c7d45fd5c5329296b633c9ed23b8475ee47b9d99")
        ).unwrap();

        assert_eq!(vk.ql_1, expected_ql);
        assert_eq!(vk.qr_1, expected_qr);
        assert_eq!(vk.qo_1, expected_qo);
        assert_eq!(vk.qm_1, expected_qm);

        assert_eq!(vk.s1_1, expected_s1);
        assert_eq!(vk.s2_1, expected_s2);
        assert_eq!(vk.s3_1, expected_s3);
    }

    // WitnessBuilder tests

    #[test]
    fn test_witness_builder_basic() {
        let system = &mut ConstraintSystem::<U64PrimeField<65537>>::new();

        let v0 = system.new_variable();
        let v1 = system.new_variable();
        let _v2 = system.add(&v0, &v1);

        // Build witness with explicit values
        let witness = WitnessBuilder::new()
            .assign(v0, FE::from(10))
            .assign(v1, FE::from(20))
            .build_with_solver(system)
            .unwrap();

        // The witness should have vectors of equal length
        assert_eq!(witness.a.len(), witness.b.len());
        assert_eq!(witness.b.len(), witness.c.len());
        // Should have at least one constraint
        assert!(witness.a.len() >= 1);
    }

    #[test]
    fn test_witness_builder_detects_missing_variable() {
        let system = &mut ConstraintSystem::<U64PrimeField<65537>>::new();

        let v0 = system.new_variable();
        let v1 = system.new_variable();
        let _v2 = system.add(&v0, &v1);

        // Try to build without providing v1
        let result = WitnessBuilder::new()
            .assign(v0, FE::from(10))
            .build(system);

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(WitnessBuilderError::MissingVariable(_))
        ));
    }

    #[test]
    fn test_witness_builder_detects_duplicate_assignment() {
        let result = WitnessBuilder::<U64PrimeField<65537>>::new()
            .set(0, FE::from(10))
            .and_then(|b| b.set(0, FE::from(20)));

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(WitnessBuilderError::DuplicateAssignment(0))
        ));
    }

    #[test]
    fn test_witness_builder_allows_overwrite_when_enabled() {
        let builder = WitnessBuilder::<U64PrimeField<65537>>::new()
            .allow_overwrite(true)
            .set(0, FE::from(10))
            .unwrap()
            .set(0, FE::from(20))
            .unwrap();

        assert_eq!(builder.assignments().get(&0), Some(&FE::from(20)));
    }

    #[test]
    fn test_witness_builder_from_assignments() {
        let mut initial = HashMap::new();
        initial.insert(0, FE::<U64PrimeField<65537>>::from(10));
        initial.insert(1, FE::from(20));

        let builder = WitnessBuilder::from_assignments(initial);
        assert_eq!(builder.assignments().len(), 2);
        assert_eq!(builder.assignments().get(&0), Some(&FE::from(10)));
    }

    #[test]
    fn test_witness_builder_set_many() {
        let assignments = vec![(0, FE::<U64PrimeField<65537>>::from(10)), (1, FE::from(20))];

        let builder = WitnessBuilder::new().set_many(assignments).unwrap();

        assert_eq!(builder.assignments().len(), 2);
    }

    #[test]
    fn test_witness_builder_with_prover() {
        use crate::prover::Prover;
        use crate::test_utils::utils::TestRandomFieldGenerator;
        use crate::verifier::Verifier;

        // Create a simple multiplication circuit: x * e == y
        let system = &mut ConstraintSystem::<FrField>::new();

        let e = system.new_variable();
        let x = system.new_public_input();
        let y = system.new_public_input();

        let z = system.mul(&x, &e);
        system.assert_eq(&y, &z);

        // Setup
        let common_preprocessed_input =
            CommonPreprocessedInput::from_constraint_system(system, &ORDER_R_MINUS_1_ROOT_UNITY)
                .unwrap();
        let srs = test_srs(common_preprocessed_input.n);
        let kzg = KZG::new(srs);
        let verifying_key = setup(&common_preprocessed_input, &kzg);

        // Build witness using WitnessBuilder with solver
        let witness = WitnessBuilder::new()
            .assign(x, FE::from(4))
            .assign(e, FE::from(3))
            .build_with_solver(system)
            .expect("Witness building failed");

        let public_inputs = vec![FE::from(4), FE::from(12)]; // x=4, y=4*3=12

        // Generate and verify proof
        let random_generator = TestRandomFieldGenerator {};
        let prover = Prover::new(kzg.clone(), random_generator);
        let proof = prover
            .prove(
                &witness,
                &public_inputs,
                &common_preprocessed_input,
                &verifying_key,
            )
            .unwrap();

        let verifier = Verifier::new(kzg);
        assert!(verifier.verify(
            &proof,
            &public_inputs,
            &common_preprocessed_input,
            &verifying_key
        ));
    }
}
