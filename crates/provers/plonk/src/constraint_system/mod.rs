pub mod conditional;
pub mod errors;
pub mod examples;
pub mod operations;
pub mod solver;
pub mod types;

use std::collections::HashMap;

use lambdaworks_math::field::{element::FieldElement, traits::IsField};

/// A constraint that enforces relations between variables. If `ConstraintType`
/// represents (Q_L, Q_R, Q_M, Q_O, Q_C), then the constraint enforces that
/// `a Q_L + b Q_R + a b Q_M + c Q_O + Q_C = 0` where `a`, `b`, and `c` are the
/// values taken by the variables `l`, `r` and `o` respectively.
#[derive(Clone)]
pub struct Constraint<F: IsField> {
    constraint_type: ConstraintType<F>,
    hint: Option<Hint<F>>,
    l: Variable,
    r: Variable,
    o: Variable,
}

/// A `ConstraintType` represents a type of gate and is determined by the values
/// of the coefficients Q_L, Q_R, Q_M, Q_O, Q_C
#[derive(Clone)]
struct ConstraintType<F: IsField> {
    ql: FieldElement<F>,
    qr: FieldElement<F>,
    qm: FieldElement<F>,
    qo: FieldElement<F>,
    qc: FieldElement<F>,
}

/// A `Column` is either `L`, `R` or `O`. It represents the role played by a
/// variable in a constraint.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Column {
    L,
    R,
    O,
}

/// A `Hint` is used to insert values to the solver. This is helpful when a
/// constraint is hard to solve but easy to check.
#[derive(Clone)]
pub struct Hint<F: IsField> {
    function: fn(&FieldElement<F>) -> FieldElement<F>,
    input: Column,
    output: Column,
}

/// Represents a variable as an ID.
pub type Variable = usize;

/// A collection of variables and constraints that encodes correct executions
/// of a program. Variables can be of two types: Public or private.
pub struct ConstraintSystem<F: IsField> {
    num_variables: usize,
    public_input_variables: Vec<Variable>,
    constraints: Vec<Constraint<F>>,
}

impl<F> ConstraintSystem<F>
where
    F: IsField,
{
    /// Returns a new empty constraint system.
    pub fn new() -> Self {
        Self {
            num_variables: 0,
            public_input_variables: Vec::new(),
            constraints: Vec::new(),
        }
    }

    /// Adds a constraint to the system.
    pub fn add_constraint(&mut self, constraint: Constraint<F>) {
        self.constraints.push(constraint);
    }

    /// Returns a null variable to be used as a placeholder
    /// in constraints.
    pub fn null_variable(&self) -> Variable {
        0
    }

    /// Creates a new variable.
    pub fn new_variable(&mut self) -> Variable {
        let variable_id = self.num_variables;
        self.num_variables += 1;
        variable_id
    }

    /// Creates a new public variable.
    pub fn new_public_input(&mut self) -> Variable {
        let new_variable = self.new_variable();
        self.public_input_variables.push(new_variable);
        new_variable
    }

    /// A dummy constraint meant to be used as padding.
    fn padding_constraint(&self) -> Constraint<F> {
        let zero = FieldElement::zero();
        Constraint {
            constraint_type: ConstraintType {
                ql: zero.clone(),
                qr: zero.clone(),
                qm: zero.clone(),
                qo: zero.clone(),
                qc: zero,
            },
            hint: None,
            l: self.null_variable(),
            r: self.null_variable(),
            o: self.null_variable(),
        }
    }

    /// Returns the public input header used in PLONK to prove the usage of the
    /// public input values.
    fn public_input_header(&self) -> Vec<Constraint<F>> {
        let zero = FieldElement::zero();
        let minus_one = -FieldElement::one();
        let mut public_input_constraints = Vec::new();
        for public_input in self.public_input_variables.iter() {
            let public_input_constraint = Constraint {
                constraint_type: ConstraintType {
                    ql: minus_one.clone(),
                    qr: zero.clone(),
                    qm: zero.clone(),
                    qo: zero.clone(),
                    qc: zero.clone(),
                },
                hint: None,
                l: *public_input,
                r: self.null_variable(),
                o: self.null_variable(),
            };
            public_input_constraints.push(public_input_constraint);
        }
        public_input_constraints
    }

    /// Returns the `LRO` and `Q` matrices. Each matrix has one row per constraint.
    /// The `LRO` matrix has 3 columns with the values of the variables IDs of every
    /// constraint. The `Q` matrix has 5 columns with the coefficients of the
    /// constraint types.
    /// Their layout is:
    /// #######################
    /// # public input header #
    /// #######################
    /// # circuit constraints #
    /// #######################
    /// #       padding       #
    /// #######################
    pub fn to_matrices(&self) -> (Vec<Variable>, Vec<FieldElement<F>>) {
        let header = self.public_input_header();
        let body = &self.constraints;
        let total_length = (header.len() + body.len()).next_power_of_two();
        let pad = vec![self.padding_constraint(); total_length - header.len() - body.len()];

        let mut full_constraints = header;
        full_constraints.extend_from_slice(body);
        full_constraints.extend_from_slice(&pad);

        let n = full_constraints.len();

        let mut lro = vec![self.null_variable(); n * 3];
        // Make a single vector with | l_1 .. l_m | r_1 .. r_m | o_1 .. o_m | concatenated.
        for (index, constraint) in full_constraints.iter().enumerate() {
            lro[index] = constraint.l;
            lro[index + n] = constraint.r;
            lro[index + n * 2] = constraint.o;
        }

        let mut q = vec![FieldElement::zero(); 5 * n];
        for (index, constraint) in full_constraints.iter().enumerate() {
            let ct = &constraint.constraint_type;
            q[index] = ct.ql.clone();
            q[index + n] = ct.qr.clone();
            q[index + 2 * n] = ct.qm.clone();
            q[index + 3 * n] = ct.qo.clone();
            q[index + 4 * n] = ct.qc.clone();
        }
        (lro, q)
    }

    /// This method filters the `values` hashmap to return the list of values
    /// corresponding to the public variables
    pub fn public_input_values(
        &self,
        values: &HashMap<Variable, FieldElement<F>>,
    ) -> Vec<FieldElement<F>> {
        let mut public_inputs = Vec::new();
        for key in &self.public_input_variables {
            if let Some(value) = values.get(key) {
                public_inputs.push(value.clone());
            }
        }
        public_inputs
    }
}

impl<F: IsField> Default for ConstraintSystem<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// This method takes the `LRO` matrix and computes the permutation used in PLONK to
/// build the copy constraint polynomial.
pub fn get_permutation(lro: &[Variable]) -> Vec<usize> {
    // For each variable store the indexes where it appears.
    let mut last_usage: HashMap<Variable, usize> = HashMap::new();
    let mut permutation = vec![0_usize; lro.len()];

    for _ in 0..2 {
        for (index, variable) in lro.iter().enumerate() {
            if last_usage.contains_key(variable) {
                permutation[index] = last_usage[variable];
            }
            last_usage.insert(*variable, index);
        }
    }

    permutation
}

#[cfg(test)]
mod tests {
    use crate::{
        prover::Prover,
        setup::{setup, CommonPreprocessedInput, Witness},
        test_utils::utils::{test_srs, TestRandomFieldGenerator, KZG, ORDER_R_MINUS_1_ROOT_UNITY},
        verifier::Verifier,
    };

    use super::*;
    use lambdaworks_math::{
        elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrField,
        field::{element::FieldElement as FE, fields::u64_prime_field::U64PrimeField},
    };

    /*
    Program:
    v0 = 1
    v1 = 2
    v2 = v0 + v1
    v3 = v1 + v0
    v4 = v2 + v3

    Variables:
    L  R  O
    0  1  2
    1  0  3
    2  3  4
    0  0  0 --> padding to next power of two

    LRO        :  0  1  2  0  1  0  3  0  2  3  4  0
    Permutation: 11  4  8  0  1  3  9  5  2  6 10  7

    */
    #[test]
    fn test_permutation() {
        let system = &mut ConstraintSystem::<U64PrimeField<65537>>::new();

        let v0 = system.new_variable();
        let v1 = system.new_variable();

        let v2 = system.add(&v0, &v1);
        let v3 = system.add(&v1, &v0);
        system.add(&v2, &v3);

        let (lro, _) = system.to_matrices();

        let permutation = get_permutation(&lro);
        let expected = vec![11, 4, 8, 0, 1, 3, 9, 5, 2, 6, 10, 7];
        assert_eq!(expected, permutation);
    }

    #[test]
    fn test_prove_simple_program_1() {
        // Program
        let system = &mut ConstraintSystem::<FrField>::new();

        let e = system.new_variable();
        let x = system.new_public_input();
        let y = system.new_public_input();

        let z = system.mul(&x, &e);
        system.assert_eq(&y, &z);

        // Common preprocessed input
        let common_preprocessed_input =
            CommonPreprocessedInput::from_constraint_system(system, &ORDER_R_MINUS_1_ROOT_UNITY);

        // Setup
        let srs = test_srs(common_preprocessed_input.n);
        let kzg = KZG::new(srs);
        let verifying_key = setup(&common_preprocessed_input, &kzg);

        // Prover:
        // 1. Generate public inputs and witness
        let inputs = HashMap::from([(x, FE::from(4)), (e, FE::from(3))]);
        let assignments = system.solve(inputs).unwrap();
        let public_inputs = system.public_input_values(&assignments);
        let witness = Witness::new(assignments, system);

        // 2. Generate proof
        let random_generator = TestRandomFieldGenerator {};
        let prover = Prover::new(kzg.clone(), random_generator);
        let proof = prover.prove(
            &witness,
            &public_inputs,
            &common_preprocessed_input,
            &verifying_key,
        );

        // Verifier
        let verifier = Verifier::new(kzg);
        assert!(verifier.verify(
            &proof,
            &public_inputs,
            &common_preprocessed_input,
            &verifying_key
        ));
    }

    #[test]
    fn test_fibonacci() {
        let system = &mut ConstraintSystem::<U64PrimeField<65537>>::new();

        let x0_initial = system.new_variable();
        let x1_initial = system.new_variable();
        let mut x0 = x0_initial;
        let mut x1 = x1_initial;

        for _ in 2..10001 {
            let x2 = system.add(&x1, &x0);
            (x0, x1) = (x1, x2);
        }

        let inputs = HashMap::from([(x0_initial, FE::from(0)), (x1_initial, FE::from(1))]);

        let expected_output = FE::from(19257);
        let assignments = system.solve(inputs).unwrap();
        assert_eq!(assignments.get(&x1).unwrap(), &expected_output);
    }
}
