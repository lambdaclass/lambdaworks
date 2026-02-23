//! Constraint System for Binius
//!
//! Binius uses a constraint system based on "shifted value indices" that
//! allows efficient representation of bitwise operations on 64-bit words.

use crate::fields::tower::Tower;

/// A variable in the constraint system
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Variable {
    /// Index of the variable
    pub index: usize,
    /// Level of the field (determines bit-width)
    pub level: usize,
}

impl Variable {
    pub fn new(index: usize, level: usize) -> Self {
        Self { index, level }
    }
}

/// Type of gate in the constraint system
#[derive(Clone, Debug)]
pub enum Gate {
    /// AND gate: output = a & b
    And {
        a: Variable,
        b: Variable,
        output: Variable,
    },
    /// Multiplication gate: output = a * b
    Mul {
        a: Variable,
        b: Variable,
        output: Variable,
    },
    /// XOR gate: output = a ^ b (derived from AND/OR)
    Xor {
        a: Variable,
        b: Variable,
        output: Variable,
    },
    /// Constant assignment
    Constant { value: Tower, output: Variable },
    /// Public input
    PublicInput { value: Tower, variable: Variable },
}

/// Constraint system for Binius
#[derive(Clone, Debug, Default)]
pub struct ConstraintSystem {
    /// List of gates/constraints
    pub gates: Vec<Gate>,
    /// Number of variables
    pub num_variables: usize,
    /// Public inputs
    pub public_inputs: Vec<Tower>,
}

impl ConstraintSystem {
    /// Create a new empty constraint system
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new variable
    pub fn new_variable(&mut self, level: usize) -> Variable {
        let var = Variable::new(self.num_variables, level);
        self.num_variables += 1;
        var
    }

    /// Add an AND gate
    pub fn AND(&mut self, a: Variable, b: Variable) -> Variable {
        let output = self.new_variable(a.level);
        self.gates.push(Gate::And { a, b, output });
        output
    }

    /// Add a multiplication gate
    pub fn MUL(&mut self, a: Variable, b: Variable) -> Variable {
        let output = self.new_variable(a.level);
        self.gates.push(Gate::Mul { a, b, output });
        output
    }

    /// Add a constant
    pub fn constant(&mut self, value: Tower) -> Variable {
        let output = self.new_variable(value.num_level());
        self.gates.push(Gate::Constant { value, output });
        output
    }

    /// Add a public input
    pub fn public_input(&mut self, value: Tower) -> Variable {
        let variable = self.new_variable(value.num_level());
        self.public_inputs.push(value);
        self.gates.push(Gate::PublicInput { value, variable });
        variable
    }

    /// Number of constraints
    pub fn num_constraints(&self) -> usize {
        self.gates.len()
    }

    /// Number of variables
    pub fn num_variables(&self) -> usize {
        self.num_variables
    }
}

/// Witness values for the constraint system
#[derive(Clone, Debug)]
pub struct Witness {
    /// Values for each variable
    pub values: Vec<Tower>,
}

impl Witness {
    pub fn new(values: Vec<Tower>) -> Self {
        Self { values }
    }

    pub fn get(&self, var: Variable) -> Tower {
        self.values[var.index]
    }
}
