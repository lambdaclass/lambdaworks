use std::collections::HashMap;

use crate::constraint_system::{get_permutation, ConstraintSystem, Variable};
use crate::prover::ProverError;
use crate::test_utils::utils::{generate_domain, generate_permutation_coefficients};
use lambdaworks_crypto::commitments::traits::IsCommitmentScheme;
use lambdaworks_crypto::fiat_shamir::{
    default_transcript::DefaultTranscript, is_transcript::IsTranscript,
};
use lambdaworks_math::errors::DeserializationError;
use lambdaworks_math::field::traits::{HasDefaultTranscript, IsFFTField};
use lambdaworks_math::field::{element::FieldElement, traits::IsField};
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::{
    deserialize_with_length, serialize_with_length, AsBytes, ByteConversion, Deserializable,
};
use sha3::{Digest, Keccak256};

/// Error types for public input handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublicInputError {
    /// Number of public inputs doesn't match the layout
    CountMismatch { expected: usize, got: usize },
    /// Public input name not found in layout
    NameNotFound(String),
    /// Duplicate public input name
    DuplicateName(String),
    /// Public input hash mismatch (layout differs between prover and verifier)
    LayoutMismatch {
        prover_hash: String,
        verifier_hash: String,
    },
}

impl std::fmt::Display for PublicInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PublicInputError::CountMismatch { expected, got } => {
                write!(
                    f,
                    "Public input count mismatch: expected {}, got {}",
                    expected, got
                )
            }
            PublicInputError::NameNotFound(name) => {
                write!(f, "Public input '{}' not found in layout", name)
            }
            PublicInputError::DuplicateName(name) => {
                write!(f, "Duplicate public input name: '{}'", name)
            }
            PublicInputError::LayoutMismatch {
                prover_hash,
                verifier_hash,
            } => {
                write!(
                    f,
                    "Public input layout mismatch: prover={}, verifier={}",
                    prover_hash, verifier_hash
                )
            }
        }
    }
}

impl std::error::Error for PublicInputError {}

/// Defines the structure and ordering of public inputs for a PLONK circuit.
///
/// Public input ordering is critical for correct verification. The prover and verifier
/// must use the exact same ordering, otherwise verification will fail. This struct
/// provides a way to define named public inputs with explicit ordering.
///
/// # Public Input Ordering Contract
///
/// Public inputs in PLONK are encoded in the first rows of the constraint system.
/// The order is determined by the sequence of `new_public_input()` calls when
/// building the circuit. This layout must match exactly between prover and verifier.
///
/// # Example
///
/// ```ignore
/// use lambdaworks_plonk::setup::PublicInputLayout;
///
/// // Define the layout with named inputs
/// let layout = PublicInputLayout::new()
///     .with_input("merkle_root")?
///     .with_input("nullifier")?
///     .with_input("amount")?;
///
/// // Get the layout hash for verification
/// let hash = layout.compute_hash();
///
/// // Build public inputs by name (order-independent)
/// let public_inputs = layout.build_inputs(&[
///     ("amount", FieldElement::from(100)),
///     ("merkle_root", merkle_root),
///     ("nullifier", nullifier),
/// ])?;
/// ```
#[derive(Clone, Debug, Default)]
pub struct PublicInputLayout {
    /// Names of public inputs in order
    names: Vec<String>,
    /// Map from name to index for fast lookup
    name_to_index: HashMap<String, usize>,
}

impl PublicInputLayout {
    /// Creates a new empty public input layout.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a named public input to the layout.
    ///
    /// The order of `with_input()` calls determines the ordering of public inputs.
    ///
    /// # Errors
    /// Returns `PublicInputError::DuplicateName` if the name already exists.
    pub fn with_input(mut self, name: &str) -> Result<Self, PublicInputError> {
        if self.name_to_index.contains_key(name) {
            return Err(PublicInputError::DuplicateName(name.to_string()));
        }
        let index = self.names.len();
        self.names.push(name.to_string());
        self.name_to_index.insert(name.to_string(), index);
        Ok(self)
    }

    /// Returns the number of public inputs in the layout.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Returns true if the layout has no public inputs.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Returns the names in order.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Returns the index of a public input by name.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.name_to_index.get(name).copied()
    }

    /// Computes a hash of the layout for comparison between prover and verifier.
    ///
    /// This hash can be included in the verification key or checked during
    /// proof generation/verification to ensure both parties use the same layout.
    pub fn compute_hash(&self) -> [u8; 32] {
        let mut hasher = Keccak256::new();
        // Hash the number of inputs
        hasher.update((self.names.len() as u64).to_be_bytes());
        // Hash each name in order
        for name in &self.names {
            hasher.update((name.len() as u32).to_be_bytes());
            hasher.update(name.as_bytes());
        }
        hasher.finalize().into()
    }

    /// Computes a hex string hash for display purposes.
    pub fn compute_hash_hex(&self) -> String {
        let hash = self.compute_hash();
        hex::encode(&hash[..8]) // First 8 bytes for readability
    }

    /// Builds a vector of public inputs from name-value pairs.
    ///
    /// The values are reordered according to the layout, so the input order
    /// of the pairs doesn't matter.
    ///
    /// # Arguments
    /// * `inputs` - Name-value pairs for public inputs
    ///
    /// # Returns
    /// * `Ok(Vec<FieldElement>)` - Public inputs in correct order
    /// * `Err(PublicInputError)` - If names don't match the layout
    pub fn build_inputs<F: IsField>(
        &self,
        inputs: &[(&str, FieldElement<F>)],
    ) -> Result<Vec<FieldElement<F>>, PublicInputError> {
        if inputs.len() != self.names.len() {
            return Err(PublicInputError::CountMismatch {
                expected: self.names.len(),
                got: inputs.len(),
            });
        }

        // Create a map from names to values
        let mut value_map: HashMap<&str, FieldElement<F>> = HashMap::new();
        for (name, value) in inputs {
            if !self.name_to_index.contains_key(*name) {
                return Err(PublicInputError::NameNotFound((*name).to_string()));
            }
            // Check for duplicate names in the input
            if value_map.insert(*name, value.clone()).is_some() {
                return Err(PublicInputError::DuplicateName((*name).to_string()));
            }
        }

        // Build the vector in layout order
        let mut result = Vec::with_capacity(self.names.len());
        for name in &self.names {
            match value_map.get(name.as_str()) {
                Some(value) => result.push(value.clone()),
                None => return Err(PublicInputError::NameNotFound(name.clone())),
            }
        }

        Ok(result)
    }

    /// Validates that the provided public inputs match the layout count.
    pub fn validate_count<F: IsField>(
        &self,
        inputs: &[FieldElement<F>],
    ) -> Result<(), PublicInputError> {
        if inputs.len() != self.names.len() {
            return Err(PublicInputError::CountMismatch {
                expected: self.names.len(),
                got: inputs.len(),
            });
        }
        Ok(())
    }

    /// Compares two layouts for equality using their hashes.
    pub fn matches(&self, other: &PublicInputLayout) -> bool {
        self.compute_hash() == other.compute_hash()
    }

    /// Verifies that this layout matches another, returning an error with details if not.
    pub fn verify_matches(&self, other: &PublicInputLayout) -> Result<(), PublicInputError> {
        if self.compute_hash() != other.compute_hash() {
            return Err(PublicInputError::LayoutMismatch {
                prover_hash: self.compute_hash_hex(),
                verifier_hash: other.compute_hash_hex(),
            });
        }
        Ok(())
    }
}

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

        let abc: Vec<_> = lro.iter().map(|v| self.assignments[v].clone()).collect();
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

/// Serialization format version for CommonPreprocessedInput.
const CPI_SERIALIZATION_VERSION: u8 = 1;

/// Helper to serialize a vector of field elements with length prefix.
fn serialize_field_element_vec<F: IsField>(elements: &[FieldElement<F>], bytes: &mut Vec<u8>)
where
    FieldElement<F>: ByteConversion,
{
    // Write length as u64
    bytes.extend_from_slice(&(elements.len() as u64).to_be_bytes());
    // Write each element with length prefix
    for elem in elements {
        let elem_bytes = elem.to_bytes_be();
        bytes.extend_from_slice(&(elem_bytes.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&elem_bytes);
    }
}

/// Helper to deserialize a vector of field elements.
fn deserialize_field_element_vec<F: IsField>(
    bytes: &[u8],
    offset: usize,
) -> Result<(usize, Vec<FieldElement<F>>), DeserializationError>
where
    FieldElement<F>: ByteConversion,
{
    if offset + 8 > bytes.len() {
        return Err(DeserializationError::InvalidAmountOfBytes);
    }

    let len = u64::from_be_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
    let mut current_offset = offset + 8;
    let mut elements = Vec::with_capacity(len);

    for _ in 0..len {
        if current_offset + 4 > bytes.len() {
            return Err(DeserializationError::InvalidAmountOfBytes);
        }
        let elem_len = u32::from_be_bytes(
            bytes[current_offset..current_offset + 4]
                .try_into()
                .unwrap(),
        ) as usize;
        current_offset += 4;

        if current_offset + elem_len > bytes.len() {
            return Err(DeserializationError::InvalidAmountOfBytes);
        }
        let elem =
            FieldElement::<F>::from_bytes_be(&bytes[current_offset..current_offset + elem_len])
                .map_err(|_| DeserializationError::FieldFromBytesError)?;
        elements.push(elem);
        current_offset += elem_len;
    }

    Ok((current_offset, elements))
}

/// Helper to serialize a polynomial (just its coefficients).
fn serialize_polynomial<F: IsField>(poly: &Polynomial<FieldElement<F>>, bytes: &mut Vec<u8>)
where
    FieldElement<F>: ByteConversion,
{
    serialize_field_element_vec(poly.coefficients(), bytes);
}

/// Helper to deserialize a polynomial.
fn deserialize_polynomial<F: IsField>(
    bytes: &[u8],
    offset: usize,
) -> Result<(usize, Polynomial<FieldElement<F>>), DeserializationError>
where
    FieldElement<F>: ByteConversion,
{
    let (new_offset, coeffs) = deserialize_field_element_vec(bytes, offset)?;
    Ok((new_offset, Polynomial::new(&coeffs)))
}

impl<F: IsFFTField> AsBytes for CommonPreprocessedInput<F>
where
    FieldElement<F>: ByteConversion,
{
    fn as_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Version byte
        bytes.push(CPI_SERIALIZATION_VERSION);

        // Domain size n (u64)
        bytes.extend_from_slice(&(self.n as u64).to_be_bytes());

        // omega and k1 (field elements with length prefix)
        let omega_bytes = self.omega.to_bytes_be();
        bytes.extend_from_slice(&(omega_bytes.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&omega_bytes);

        let k1_bytes = self.k1.to_bytes_be();
        bytes.extend_from_slice(&(k1_bytes.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&k1_bytes);

        // Selector polynomials (coefficient form)
        serialize_polynomial(&self.ql, &mut bytes);
        serialize_polynomial(&self.qr, &mut bytes);
        serialize_polynomial(&self.qo, &mut bytes);
        serialize_polynomial(&self.qm, &mut bytes);
        serialize_polynomial(&self.qc, &mut bytes);

        // Permutation polynomials (coefficient form)
        serialize_polynomial(&self.s1, &mut bytes);
        serialize_polynomial(&self.s2, &mut bytes);
        serialize_polynomial(&self.s3, &mut bytes);

        // Lagrange form vectors (for efficiency, avoid recomputation)
        serialize_field_element_vec(&self.s1_lagrange, &mut bytes);
        serialize_field_element_vec(&self.s2_lagrange, &mut bytes);
        serialize_field_element_vec(&self.s3_lagrange, &mut bytes);

        bytes
    }
}

impl<F: IsFFTField> Deserializable for CommonPreprocessedInput<F>
where
    FieldElement<F>: ByteConversion,
{
    fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError>
    where
        Self: Sized,
    {
        if bytes.is_empty() {
            return Err(DeserializationError::InvalidAmountOfBytes);
        }

        // Check version
        let version = bytes[0];
        if version != CPI_SERIALIZATION_VERSION {
            return Err(DeserializationError::InvalidValue);
        }

        let mut offset = 1;

        // Read n
        if offset + 8 > bytes.len() {
            return Err(DeserializationError::InvalidAmountOfBytes);
        }
        let n = u64::from_be_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        // Read omega
        if offset + 4 > bytes.len() {
            return Err(DeserializationError::InvalidAmountOfBytes);
        }
        let omega_len = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        if offset + omega_len > bytes.len() {
            return Err(DeserializationError::InvalidAmountOfBytes);
        }
        let omega = FieldElement::<F>::from_bytes_be(&bytes[offset..offset + omega_len])
            .map_err(|_| DeserializationError::FieldFromBytesError)?;
        offset += omega_len;

        // Read k1
        if offset + 4 > bytes.len() {
            return Err(DeserializationError::InvalidAmountOfBytes);
        }
        let k1_len = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        if offset + k1_len > bytes.len() {
            return Err(DeserializationError::InvalidAmountOfBytes);
        }
        let k1 = FieldElement::<F>::from_bytes_be(&bytes[offset..offset + k1_len])
            .map_err(|_| DeserializationError::FieldFromBytesError)?;
        offset += k1_len;

        // Read selector polynomials
        let (new_offset, ql) = deserialize_polynomial(bytes, offset)?;
        offset = new_offset;
        let (new_offset, qr) = deserialize_polynomial(bytes, offset)?;
        offset = new_offset;
        let (new_offset, qo) = deserialize_polynomial(bytes, offset)?;
        offset = new_offset;
        let (new_offset, qm) = deserialize_polynomial(bytes, offset)?;
        offset = new_offset;
        let (new_offset, qc) = deserialize_polynomial(bytes, offset)?;
        offset = new_offset;

        // Read permutation polynomials
        let (new_offset, s1) = deserialize_polynomial(bytes, offset)?;
        offset = new_offset;
        let (new_offset, s2) = deserialize_polynomial(bytes, offset)?;
        offset = new_offset;
        let (new_offset, s3) = deserialize_polynomial(bytes, offset)?;
        offset = new_offset;

        // Read Lagrange form vectors
        let (new_offset, s1_lagrange) = deserialize_field_element_vec(bytes, offset)?;
        offset = new_offset;
        let (new_offset, s2_lagrange) = deserialize_field_element_vec(bytes, offset)?;
        offset = new_offset;
        let (_, s3_lagrange) = deserialize_field_element_vec(bytes, offset)?;

        // Regenerate domain from omega and n
        let domain = generate_domain(&omega, n);

        Ok(CommonPreprocessedInput {
            n,
            domain,
            omega,
            k1,
            ql,
            qr,
            qo,
            qm,
            qc,
            s1,
            s2,
            s3,
            s1_lagrange,
            s2_lagrange,
            s3_lagrange,
        })
    }
}

/// PLONK verification key containing commitments to selector and permutation polynomials.
///
/// The verification key is derived from the circuit's constraint system during setup
/// and is used by the verifier to check proofs.
#[derive(Clone, Debug, PartialEq, Eq)]
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

/// Serialization format version for VerificationKey.
/// Increment this when making breaking changes to the serialization format.
const VK_SERIALIZATION_VERSION: u8 = 1;

impl<G1Point> AsBytes for VerificationKey<G1Point>
where
    G1Point: AsBytes,
{
    fn as_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Version byte for forward compatibility
        bytes.push(VK_SERIALIZATION_VERSION);

        // Serialize all 8 commitments in a consistent order
        // Order: qm, ql, qr, qo, qc, s1, s2, s3
        for commitment in [
            &self.qm_1, &self.ql_1, &self.qr_1, &self.qo_1, &self.qc_1, &self.s1_1, &self.s2_1,
            &self.s3_1,
        ] {
            bytes.extend(serialize_with_length(commitment));
        }

        bytes
    }
}

impl<G1Point> Deserializable for VerificationKey<G1Point>
where
    G1Point: Deserializable,
{
    fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError>
    where
        Self: Sized,
    {
        if bytes.is_empty() {
            return Err(DeserializationError::InvalidAmountOfBytes);
        }

        // Check version (use InvalidValue for version mismatch)
        let version = bytes[0];
        if version != VK_SERIALIZATION_VERSION {
            return Err(DeserializationError::InvalidValue);
        }

        let mut offset = 1; // Skip version byte

        let (new_offset, qm_1) = deserialize_with_length(bytes, offset)?;
        offset = new_offset;
        let (new_offset, ql_1) = deserialize_with_length(bytes, offset)?;
        offset = new_offset;
        let (new_offset, qr_1) = deserialize_with_length(bytes, offset)?;
        offset = new_offset;
        let (new_offset, qo_1) = deserialize_with_length(bytes, offset)?;
        offset = new_offset;
        let (new_offset, qc_1) = deserialize_with_length(bytes, offset)?;
        offset = new_offset;
        let (new_offset, s1_1) = deserialize_with_length(bytes, offset)?;
        offset = new_offset;
        let (new_offset, s2_1) = deserialize_with_length(bytes, offset)?;
        offset = new_offset;
        let (_, s3_1) = deserialize_with_length(bytes, offset)?;

        Ok(VerificationKey {
            qm_1,
            ql_1,
            qr_1,
            qo_1,
            qc_1,
            s1_1,
            s2_1,
            s3_1,
        })
    }
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
        assert!(!witness.a.is_empty());
    }

    #[test]
    fn test_witness_builder_detects_missing_variable() {
        let system = &mut ConstraintSystem::<U64PrimeField<65537>>::new();

        let v0 = system.new_variable();
        let v1 = system.new_variable();
        let _v2 = system.add(&v0, &v1);

        // Try to build without providing v1
        let result = WitnessBuilder::new().assign(v0, FE::from(10)).build(system);

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

    // VerificationKey serialization tests

    #[test]
    fn test_verification_key_serialization_roundtrip() {
        let common_input = test_common_preprocessed_input_1();
        let srs = test_srs(common_input.n);
        let kzg = KZG::new(srs);

        let vk = setup::<FrField, KZG>(&common_input, &kzg);

        // Serialize
        let serialized = vk.as_bytes();

        // Check version byte
        assert_eq!(serialized[0], 1, "Version byte should be 1");

        // Deserialize
        use crate::test_utils::utils::G1Point;
        let deserialized: VerificationKey<G1Point> =
            VerificationKey::deserialize(&serialized).expect("Deserialization failed");

        // Verify all fields match
        assert_eq!(vk.qm_1, deserialized.qm_1);
        assert_eq!(vk.ql_1, deserialized.ql_1);
        assert_eq!(vk.qr_1, deserialized.qr_1);
        assert_eq!(vk.qo_1, deserialized.qo_1);
        assert_eq!(vk.qc_1, deserialized.qc_1);
        assert_eq!(vk.s1_1, deserialized.s1_1);
        assert_eq!(vk.s2_1, deserialized.s2_1);
        assert_eq!(vk.s3_1, deserialized.s3_1);
    }

    #[test]
    fn test_verification_key_deserialization_empty_bytes() {
        use crate::test_utils::utils::G1Point;
        let result: Result<VerificationKey<G1Point>, _> = VerificationKey::deserialize(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_verification_key_deserialization_wrong_version() {
        let common_input = test_common_preprocessed_input_1();
        let srs = test_srs(common_input.n);
        let kzg = KZG::new(srs);
        let vk = setup::<FrField, KZG>(&common_input, &kzg);

        let mut serialized = vk.as_bytes();
        // Modify version byte to invalid version
        serialized[0] = 99;

        use crate::test_utils::utils::G1Point;
        let result: Result<VerificationKey<G1Point>, _> = VerificationKey::deserialize(&serialized);
        assert!(result.is_err());
    }

    // CommonPreprocessedInput serialization tests

    #[test]
    fn test_common_preprocessed_input_serialization_roundtrip() {
        let common_input = test_common_preprocessed_input_1();

        // Serialize
        let serialized = common_input.as_bytes();

        // Check version byte
        assert_eq!(serialized[0], 1, "Version byte should be 1");

        // Deserialize
        let deserialized: CommonPreprocessedInput<FrField> =
            CommonPreprocessedInput::deserialize(&serialized).expect("Deserialization failed");

        // Verify core fields
        assert_eq!(common_input.n, deserialized.n);
        assert_eq!(common_input.omega, deserialized.omega);
        assert_eq!(common_input.k1, deserialized.k1);

        // Verify domain is correctly regenerated
        assert_eq!(common_input.domain.len(), deserialized.domain.len());
        for (a, b) in common_input.domain.iter().zip(deserialized.domain.iter()) {
            assert_eq!(a, b);
        }

        // Verify selector polynomials
        assert_eq!(
            common_input.ql.coefficients(),
            deserialized.ql.coefficients()
        );
        assert_eq!(
            common_input.qr.coefficients(),
            deserialized.qr.coefficients()
        );
        assert_eq!(
            common_input.qo.coefficients(),
            deserialized.qo.coefficients()
        );
        assert_eq!(
            common_input.qm.coefficients(),
            deserialized.qm.coefficients()
        );
        assert_eq!(
            common_input.qc.coefficients(),
            deserialized.qc.coefficients()
        );

        // Verify permutation polynomials
        assert_eq!(
            common_input.s1.coefficients(),
            deserialized.s1.coefficients()
        );
        assert_eq!(
            common_input.s2.coefficients(),
            deserialized.s2.coefficients()
        );
        assert_eq!(
            common_input.s3.coefficients(),
            deserialized.s3.coefficients()
        );

        // Verify Lagrange form vectors
        assert_eq!(common_input.s1_lagrange, deserialized.s1_lagrange);
        assert_eq!(common_input.s2_lagrange, deserialized.s2_lagrange);
        assert_eq!(common_input.s3_lagrange, deserialized.s3_lagrange);
    }

    #[test]
    fn test_common_preprocessed_input_deserialization_empty_bytes() {
        let result: Result<CommonPreprocessedInput<FrField>, _> =
            CommonPreprocessedInput::deserialize(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_common_preprocessed_input_deserialization_wrong_version() {
        let common_input = test_common_preprocessed_input_1();
        let mut serialized = common_input.as_bytes();

        // Modify version byte to invalid version
        serialized[0] = 99;

        let result: Result<CommonPreprocessedInput<FrField>, _> =
            CommonPreprocessedInput::deserialize(&serialized);
        assert!(result.is_err());
    }

    #[test]
    fn test_common_preprocessed_input_proof_still_verifies_after_roundtrip() {
        use crate::prover::Prover;
        use crate::test_utils::utils::TestRandomFieldGenerator;
        use crate::verifier::Verifier;

        // Create circuit and common preprocessed input
        let common_input = test_common_preprocessed_input_1();

        // Serialize and deserialize
        let serialized = common_input.as_bytes();
        let deserialized: CommonPreprocessedInput<FrField> =
            CommonPreprocessedInput::deserialize(&serialized).expect("Deserialization failed");

        // Setup with deserialized input
        let srs = test_srs(deserialized.n);
        let kzg = KZG::new(srs);
        let vk = setup(&deserialized, &kzg);

        // Create witness
        use crate::test_utils::circuit_1::test_witness_1;
        let x = FE::from(4_u64);
        let y = FE::from(12_u64);
        let e = FE::from(3_u64);
        let public_input = vec![x.clone(), y];
        let witness = test_witness_1(x, e);

        // Prove and verify
        let random_generator = TestRandomFieldGenerator {};
        let prover = Prover::new(kzg.clone(), random_generator);
        let proof = prover
            .prove(&witness, &public_input, &deserialized, &vk)
            .unwrap();

        let verifier = Verifier::new(kzg);
        assert!(verifier.verify(&proof, &public_input, &deserialized, &vk));
    }

    // PublicInputLayout tests

    #[test]
    fn test_public_input_layout_basic() {
        let layout = PublicInputLayout::new()
            .with_input("x")
            .unwrap()
            .with_input("y")
            .unwrap();

        assert_eq!(layout.len(), 2);
        assert_eq!(layout.names(), &["x", "y"]);
        assert_eq!(layout.index_of("x"), Some(0));
        assert_eq!(layout.index_of("y"), Some(1));
        assert_eq!(layout.index_of("z"), None);
    }

    #[test]
    fn test_public_input_layout_rejects_duplicate() {
        let result = PublicInputLayout::new()
            .with_input("x")
            .unwrap()
            .with_input("x");

        assert!(result.is_err());
        assert!(matches!(result, Err(PublicInputError::DuplicateName(_))));
    }

    #[test]
    fn test_public_input_layout_build_inputs() {
        let layout = PublicInputLayout::new()
            .with_input("a")
            .unwrap()
            .with_input("b")
            .unwrap()
            .with_input("c")
            .unwrap();

        // Build in different order than layout
        let inputs = layout
            .build_inputs(&[
                ("c", FE::<FrField>::from(3_u64)),
                ("a", FE::from(1_u64)),
                ("b", FE::from(2_u64)),
            ])
            .unwrap();

        // Should be reordered to layout order: a, b, c
        assert_eq!(inputs[0], FE::from(1_u64));
        assert_eq!(inputs[1], FE::from(2_u64));
        assert_eq!(inputs[2], FE::from(3_u64));
    }

    #[test]
    fn test_public_input_layout_build_inputs_wrong_count() {
        let layout = PublicInputLayout::new()
            .with_input("a")
            .unwrap()
            .with_input("b")
            .unwrap();

        let result = layout.build_inputs::<FrField>(&[("a", FE::from(1_u64))]);

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(PublicInputError::CountMismatch {
                expected: 2,
                got: 1
            })
        ));
    }

    #[test]
    fn test_public_input_layout_build_inputs_wrong_name() {
        let layout = PublicInputLayout::new()
            .with_input("a")
            .unwrap()
            .with_input("b")
            .unwrap();

        let result = layout.build_inputs(&[
            ("a", FE::<FrField>::from(1_u64)),
            ("c", FE::from(2_u64)), // Wrong name
        ]);

        assert!(result.is_err());
        assert!(matches!(result, Err(PublicInputError::NameNotFound(_))));
    }

    #[test]
    fn test_public_input_layout_build_inputs_rejects_duplicate_in_call() {
        let layout = PublicInputLayout::new()
            .with_input("a")
            .unwrap()
            .with_input("b")
            .unwrap()
            .with_input("c")
            .unwrap();

        // Attempt to provide duplicate "a" in the same build_inputs call
        let result = layout.build_inputs(&[
            ("a", FE::<FrField>::from(1_u64)),
            ("b", FE::from(2_u64)),
            ("a", FE::from(99_u64)), // Duplicate name
        ]);

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(PublicInputError::DuplicateName(ref name)) if name == "a"
        ));
    }

    #[test]
    fn test_public_input_layout_hash_deterministic() {
        let layout1 = PublicInputLayout::new()
            .with_input("x")
            .unwrap()
            .with_input("y")
            .unwrap();

        let layout2 = PublicInputLayout::new()
            .with_input("x")
            .unwrap()
            .with_input("y")
            .unwrap();

        assert_eq!(layout1.compute_hash(), layout2.compute_hash());
        assert!(layout1.matches(&layout2));
    }

    #[test]
    fn test_public_input_layout_hash_differs_by_order() {
        let layout1 = PublicInputLayout::new()
            .with_input("x")
            .unwrap()
            .with_input("y")
            .unwrap();

        let layout2 = PublicInputLayout::new()
            .with_input("y")
            .unwrap()
            .with_input("x")
            .unwrap();

        assert_ne!(layout1.compute_hash(), layout2.compute_hash());
        assert!(!layout1.matches(&layout2));
    }

    #[test]
    fn test_public_input_layout_hash_differs_by_name() {
        let layout1 = PublicInputLayout::new()
            .with_input("x")
            .unwrap()
            .with_input("y")
            .unwrap();

        let layout2 = PublicInputLayout::new()
            .with_input("x")
            .unwrap()
            .with_input("z")
            .unwrap();

        assert_ne!(layout1.compute_hash(), layout2.compute_hash());
        assert!(!layout1.matches(&layout2));
    }

    #[test]
    fn test_public_input_layout_verify_matches() {
        let layout1 = PublicInputLayout::new()
            .with_input("x")
            .unwrap()
            .with_input("y")
            .unwrap();

        let layout2 = PublicInputLayout::new()
            .with_input("x")
            .unwrap()
            .with_input("y")
            .unwrap();

        assert!(layout1.verify_matches(&layout2).is_ok());

        let layout3 = PublicInputLayout::new()
            .with_input("a")
            .unwrap()
            .with_input("b")
            .unwrap();

        let result = layout1.verify_matches(&layout3);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(PublicInputError::LayoutMismatch { .. })
        ));
    }

    #[test]
    fn test_public_input_layout_empty() {
        let layout = PublicInputLayout::new();
        assert!(layout.is_empty());
        assert_eq!(layout.len(), 0);

        // Building with empty layout should work with empty inputs
        let inputs = layout.build_inputs::<FrField>(&[]).unwrap();
        assert!(inputs.is_empty());
    }
}
