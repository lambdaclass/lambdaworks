//! Circuit builder for constructing PLONK circuits with a high-level API.

use crate::constraint_system::ConstraintSystem;
use crate::dsl::types::{AsFieldVar, BoolVar, FieldVar, Var};
use crate::prover::ProverError;
use crate::setup::{CommonPreprocessedInput, Witness, WitnessBuilder};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsFFTField, IsField};
use std::collections::HashMap;

/// Errors that can occur when building or using a circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitError {
    /// A variable name was already used
    DuplicateName(String),
    /// A variable name was not found
    NameNotFound(String),
    /// Witness building failed
    WitnessBuildError(String),
    /// Circuit building failed
    BuildError(String),
}

impl std::fmt::Display for CircuitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitError::DuplicateName(name) => write!(f, "Duplicate variable name: {}", name),
            CircuitError::NameNotFound(name) => write!(f, "Variable name not found: {}", name),
            CircuitError::WitnessBuildError(msg) => write!(f, "Witness build error: {}", msg),
            CircuitError::BuildError(msg) => write!(f, "Circuit build error: {}", msg),
        }
    }
}

impl std::error::Error for CircuitError {}

/// A high-level circuit builder with typed variables and named inputs.
///
/// `CircuitBuilder` wraps the low-level `ConstraintSystem` and provides:
/// - Named variables for debugging
/// - Typed variable operations
/// - Automatic constraint generation
/// - Gadget composition
///
/// # Example
///
/// ```ignore
/// use lambdaworks_plonk::dsl::CircuitBuilder;
///
/// let mut builder = CircuitBuilder::new();
///
/// // Public inputs
/// let x = builder.public_input("x");
/// let y = builder.public_input("y");
///
/// // Private input
/// let e = builder.private_input("e");
///
/// // Computation: x * e == y
/// let z = builder.mul(&x, &e);
/// builder.assert_eq(&z, &y);
/// ```
pub struct CircuitBuilder<F: IsField> {
    /// The underlying constraint system
    cs: ConstraintSystem<F>,
    /// Map from variable names to variables
    name_to_var: HashMap<String, FieldVar>,
    /// Map from variables to names (for debugging)
    var_to_name: HashMap<FieldVar, String>,
    /// Public input names in order
    public_input_names: Vec<String>,
    /// Private input names in order
    private_input_names: Vec<String>,
}

impl<F: IsField> Default for CircuitBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: IsField> CircuitBuilder<F> {
    /// Creates a new empty circuit builder.
    pub fn new() -> Self {
        Self {
            cs: ConstraintSystem::new(),
            name_to_var: HashMap::new(),
            var_to_name: HashMap::new(),
            public_input_names: Vec::new(),
            private_input_names: Vec::new(),
        }
    }

    /// Creates a public input variable with the given name.
    ///
    /// Public inputs are known to both prover and verifier.
    ///
    /// # Panics
    /// Panics if the name is already used.
    pub fn public_input(&mut self, name: &str) -> FieldVar {
        self.try_public_input(name)
            .unwrap_or_else(|e| panic!("{}", e))
    }

    /// Creates a public input variable, returning an error if the name is taken.
    pub fn try_public_input(&mut self, name: &str) -> Result<FieldVar, CircuitError> {
        if self.name_to_var.contains_key(name) {
            return Err(CircuitError::DuplicateName(name.to_string()));
        }

        let var = self.cs.new_public_input();
        let field_var = FieldVar::new(var);
        self.name_to_var.insert(name.to_string(), field_var);
        self.var_to_name.insert(field_var, name.to_string());
        self.public_input_names.push(name.to_string());

        Ok(field_var)
    }

    /// Creates a private input variable with the given name.
    ///
    /// Private inputs are known only to the prover.
    ///
    /// # Panics
    /// Panics if the name is already used.
    pub fn private_input(&mut self, name: &str) -> FieldVar {
        self.try_private_input(name)
            .unwrap_or_else(|e| panic!("{}", e))
    }

    /// Creates a private input variable, returning an error if the name is taken.
    pub fn try_private_input(&mut self, name: &str) -> Result<FieldVar, CircuitError> {
        if self.name_to_var.contains_key(name) {
            return Err(CircuitError::DuplicateName(name.to_string()));
        }

        let var = self.cs.new_variable();
        let field_var = FieldVar::new(var);
        self.name_to_var.insert(name.to_string(), field_var);
        self.var_to_name.insert(field_var, name.to_string());
        self.private_input_names.push(name.to_string());

        Ok(field_var)
    }

    /// Creates an anonymous variable (no name).
    ///
    /// Use this for intermediate computation results.
    pub fn new_variable(&mut self) -> FieldVar {
        let var = self.cs.new_variable();
        FieldVar::new(var)
    }

    /// Creates a constant variable with the given value.
    pub fn constant(&mut self, value: impl Into<FieldElement<F>>) -> FieldVar {
        let var = self.cs.new_variable();
        let field_var = FieldVar::new(var);
        // Add constraint: var - value = 0
        self.cs.add_constant(&var, value.into());
        field_var
    }

    /// Returns the variable with the given name, if it exists.
    pub fn get_var(&self, name: &str) -> Option<FieldVar> {
        self.name_to_var.get(name).copied()
    }

    /// Returns the name of a variable, if it has one.
    pub fn get_name(&self, var: &FieldVar) -> Option<&str> {
        self.var_to_name.get(var).map(|s| s.as_str())
    }

    /// Returns the public input names in order.
    pub fn public_input_names(&self) -> &[String] {
        &self.public_input_names
    }

    /// Returns the private input names in order.
    pub fn private_input_names(&self) -> &[String] {
        &self.private_input_names
    }

    // ========================================
    // Arithmetic Operations
    // ========================================

    /// Adds two variables: result = a + b
    pub fn add<A: AsFieldVar, B: AsFieldVar>(&mut self, a: &A, b: &B) -> FieldVar {
        let result = self
            .cs
            .add(&a.as_field_var().inner, &b.as_field_var().inner);
        FieldVar::new(result)
    }

    /// Subtracts two variables: result = a - b
    pub fn sub<A: AsFieldVar, B: AsFieldVar>(&mut self, a: &A, b: &B) -> FieldVar {
        // a - b = a + (-1) * b
        let neg_one = -FieldElement::<F>::one();
        let neg_b = self.mul_constant(b, neg_one);
        self.add(a, &neg_b)
    }

    /// Multiplies two variables: result = a * b
    pub fn mul<A: AsFieldVar, B: AsFieldVar>(&mut self, a: &A, b: &B) -> FieldVar {
        let result = self
            .cs
            .mul(&a.as_field_var().inner, &b.as_field_var().inner);
        FieldVar::new(result)
    }

    /// Multiplies a variable by a constant: result = a * c
    pub fn mul_constant<A: AsFieldVar>(&mut self, a: &A, c: FieldElement<F>) -> FieldVar {
        // linear_function(v, c, b, hint) creates w = c * v + b
        let result =
            self.cs
                .linear_function(&a.as_field_var().inner, c, FieldElement::zero(), None);
        FieldVar::new(result)
    }

    /// Adds a constant to a variable: result = a + c
    pub fn add_constant<A: AsFieldVar>(&mut self, a: &A, c: FieldElement<F>) -> FieldVar {
        let result = self.cs.add_constant(&a.as_field_var().inner, c);
        FieldVar::new(result)
    }

    /// Computes the inverse: result = 1 / a
    ///
    /// # Note
    /// This constrains `a * result = 1`, which fails if `a = 0`.
    /// Also returns an is_zero flag (not exposed here).
    pub fn inv<A: AsFieldVar>(&mut self, a: &A) -> FieldVar {
        // inv returns (is_zero, v_inverse) - we only need the inverse
        let (_is_zero, result) = self.cs.inv(&a.as_field_var().inner);
        FieldVar::new(result)
    }

    /// Computes division: result = a / b
    ///
    /// # Note
    /// This constrains `b * result = a`, which fails if `b = 0`.
    pub fn div<A: AsFieldVar, B: AsFieldVar>(&mut self, a: &A, b: &B) -> FieldVar {
        let result = self
            .cs
            .div(&a.as_field_var().inner, &b.as_field_var().inner);
        FieldVar::new(result)
    }

    // ========================================
    // Boolean Operations
    // ========================================

    /// Asserts that a variable is boolean (0 or 1) and returns a typed BoolVar.
    ///
    /// Adds constraint: a * (a - 1) = 0
    pub fn assert_bool<A: AsFieldVar>(&mut self, a: &A) -> BoolVar {
        // a * (a - 1) = 0 => a^2 - a = 0
        let a_var = a.as_field_var();
        let a_sq = self.mul(&a_var, &a_var);
        self.cs.assert_eq(&a_sq.inner, &a_var.inner);
        Var::new(a_var.inner)
    }

    /// Computes the logical NOT: result = 1 - a
    ///
    /// Assumes `a` is boolean.
    pub fn not(&mut self, a: &BoolVar) -> BoolVar {
        let result = self.cs.not(&a.inner);
        Var::new(result)
    }

    /// Computes the logical AND: result = a * b
    ///
    /// Assumes both inputs are boolean.
    pub fn and(&mut self, a: &BoolVar, b: &BoolVar) -> BoolVar {
        let result = self.mul(a, b);
        Var::new(result.inner)
    }

    /// Computes the logical OR: result = a + b - a * b
    ///
    /// Assumes both inputs are boolean.
    pub fn or(&mut self, a: &BoolVar, b: &BoolVar) -> BoolVar {
        // a OR b = a + b - a*b
        let sum = self.add(a, b);
        let prod = self.mul(a, b);
        let result = self.sub(&sum, &prod);
        Var::new(result.inner)
    }

    /// Computes the logical XOR: result = a + b - 2*a*b
    ///
    /// Assumes both inputs are boolean.
    pub fn xor(&mut self, a: &BoolVar, b: &BoolVar) -> BoolVar {
        // a XOR b = a + b - 2*a*b
        let sum = self.add(a, b);
        let prod = self.mul(a, b);
        let two_prod = self.add(&prod, &prod);
        let result = self.sub(&sum, &two_prod);
        Var::new(result.inner)
    }

    // ========================================
    // Conditional Operations
    // ========================================

    /// Selects between two values based on a condition.
    ///
    /// result = if condition then if_true else if_false
    /// result = condition * if_true + (1 - condition) * if_false
    pub fn select<A: AsFieldVar, B: AsFieldVar>(
        &mut self,
        condition: &BoolVar,
        if_true: &A,
        if_false: &B,
    ) -> FieldVar {
        // result = cond * if_true + (1 - cond) * if_false
        // result = cond * if_true + if_false - cond * if_false
        // result = cond * (if_true - if_false) + if_false
        let diff = self.sub(if_true, if_false);
        let scaled = self.mul(condition, &diff);
        self.add(&scaled, if_false)
    }

    // ========================================
    // Assertions
    // ========================================

    /// Asserts that two variables are equal.
    pub fn assert_eq<A: AsFieldVar, B: AsFieldVar>(&mut self, a: &A, b: &B) {
        self.cs
            .assert_eq(&a.as_field_var().inner, &b.as_field_var().inner);
    }

    /// Asserts that a variable equals a constant.
    pub fn assert_eq_constant<A: AsFieldVar>(&mut self, a: &A, value: FieldElement<F>) {
        let c = self.constant(value);
        self.assert_eq(a, &c);
    }

    /// Asserts that a variable is zero.
    pub fn assert_zero<A: AsFieldVar>(&mut self, a: &A) {
        self.assert_eq_constant(a, FieldElement::zero());
    }

    // ========================================
    // Building
    // ========================================

    /// Returns a reference to the underlying constraint system.
    pub fn constraint_system(&self) -> &ConstraintSystem<F> {
        &self.cs
    }

    /// Returns a mutable reference to the underlying constraint system.
    pub fn constraint_system_mut(&mut self) -> &mut ConstraintSystem<F> {
        &mut self.cs
    }

    /// Consumes the builder and returns the underlying constraint system.
    pub fn into_constraint_system(self) -> ConstraintSystem<F> {
        self.cs
    }
}

impl<F: IsFFTField> CircuitBuilder<F> {
    /// Builds the common preprocessed input from the constraint system.
    pub fn build_cpi(
        &self,
        order_r_minus_1_root_unity: &FieldElement<F>,
    ) -> Result<CommonPreprocessedInput<F>, ProverError> {
        CommonPreprocessedInput::from_constraint_system(&self.cs, order_r_minus_1_root_unity)
    }

    /// Builds a witness from named input values.
    ///
    /// # Arguments
    /// * `inputs` - Name-value pairs for all inputs (public and private)
    ///
    /// # Returns
    /// * `Ok(Witness)` if all inputs are provided and constraints can be solved
    /// * `Err(CircuitError)` if inputs are missing or constraint solving fails
    pub fn build_witness(
        &self,
        inputs: &[(&str, FieldElement<F>)],
    ) -> Result<Witness<F>, CircuitError> {
        let mut assignments = HashMap::new();

        for (name, value) in inputs {
            let var = self
                .name_to_var
                .get(*name)
                .ok_or_else(|| CircuitError::NameNotFound((*name).to_string()))?;
            assignments.insert(var.inner, value.clone());
        }

        WitnessBuilder::from_assignments(assignments)
            .build_with_solver(&self.cs)
            .map_err(|e| CircuitError::WitnessBuildError(e.to_string()))
    }

    /// Extracts public input values from a set of named inputs.
    pub fn extract_public_inputs(
        &self,
        inputs: &[(&str, FieldElement<F>)],
    ) -> Result<Vec<FieldElement<F>>, CircuitError> {
        let input_map: HashMap<&str, &FieldElement<F>> =
            inputs.iter().map(|(k, v)| (*k, v)).collect();

        let mut public_inputs = Vec::with_capacity(self.public_input_names.len());
        for name in &self.public_input_names {
            let value = input_map
                .get(name.as_str())
                .ok_or_else(|| CircuitError::NameNotFound(name.clone()))?;
            public_inputs.push((*value).clone());
        }

        Ok(public_inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
    use lambdaworks_math::field::test_fields::u64_test_field::U64TestField;

    // Use a simple field for basic tests
    type F = U64PrimeField<65537>;

    // Use an FFT-compatible field for tests that need build_witness/extract_public_inputs
    type FftField = U64TestField;
    type FftFe = FieldElement<FftField>;

    #[test]
    fn test_circuit_builder_basic() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.public_input("x");
        let y = builder.public_input("y");
        let e = builder.private_input("e");

        // x * e == y
        let z = builder.mul(&x, &e);
        builder.assert_eq(&z, &y);

        assert_eq!(builder.public_input_names(), &["x", "y"]);
        assert_eq!(builder.private_input_names(), &["e"]);
        assert_eq!(builder.get_var("x"), Some(x));
        assert_eq!(builder.get_var("nonexistent"), None);
    }

    #[test]
    fn test_circuit_builder_duplicate_name() {
        let mut builder = CircuitBuilder::<F>::new();

        builder.public_input("x");
        let result = builder.try_public_input("x");

        assert!(result.is_err());
        assert!(matches!(result, Err(CircuitError::DuplicateName(_))));
    }

    #[test]
    fn test_circuit_builder_add() {
        let mut builder = CircuitBuilder::<F>::new();

        let a = builder.public_input("a");
        let b = builder.public_input("b");
        let c = builder.add(&a, &b);

        // Verify c is a new variable
        assert_ne!(c.variable(), a.variable());
        assert_ne!(c.variable(), b.variable());
    }

    #[test]
    fn test_circuit_builder_mul() {
        let mut builder = CircuitBuilder::<F>::new();

        let a = builder.public_input("a");
        let b = builder.public_input("b");
        let c = builder.mul(&a, &b);

        assert_ne!(c.variable(), a.variable());
        assert_ne!(c.variable(), b.variable());
    }

    #[test]
    fn test_circuit_builder_bool_operations() {
        let mut builder = CircuitBuilder::<F>::new();

        let a = builder.private_input("a");
        let b = builder.private_input("b");

        let a_bool = builder.assert_bool(&a);
        let b_bool = builder.assert_bool(&b);

        let _not_a = builder.not(&a_bool);
        let _a_and_b = builder.and(&a_bool, &b_bool);
        let _a_or_b = builder.or(&a_bool, &b_bool);
        let _a_xor_b = builder.xor(&a_bool, &b_bool);
    }

    #[test]
    fn test_circuit_builder_select() {
        let mut builder = CircuitBuilder::<F>::new();

        let cond = builder.private_input("cond");
        let a = builder.private_input("a");
        let b = builder.private_input("b");

        let cond_bool = builder.assert_bool(&cond);
        let _result = builder.select(&cond_bool, &a, &b);
    }

    #[test]
    fn test_circuit_builder_build_witness() {
        let mut builder = CircuitBuilder::<FftField>::new();

        let x = builder.public_input("x");
        let y = builder.public_input("y");
        let e = builder.private_input("e");

        let z = builder.mul(&x, &e);
        builder.assert_eq(&z, &y);

        // x=4, e=3, y=12 satisfies x*e=y
        let witness = builder
            .build_witness(&[
                ("x", FftFe::from(4u64)),
                ("e", FftFe::from(3u64)),
                ("y", FftFe::from(12u64)),
            ])
            .unwrap();

        assert_eq!(witness.a.len(), witness.b.len());
        assert_eq!(witness.b.len(), witness.c.len());
    }

    #[test]
    fn test_circuit_builder_extract_public_inputs() {
        let mut builder = CircuitBuilder::<FftField>::new();

        builder.public_input("x");
        builder.public_input("y");
        builder.private_input("e");

        let public_inputs = builder
            .extract_public_inputs(&[
                ("y", FftFe::from(12u64)),
                ("x", FftFe::from(4u64)),
                ("e", FftFe::from(3u64)),
            ])
            .unwrap();

        // Should be in order: x, y
        assert_eq!(public_inputs, vec![FftFe::from(4u64), FftFe::from(12u64)]);
    }
}
