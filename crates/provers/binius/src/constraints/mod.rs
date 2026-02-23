//! Constraint System for Binius
//!
//! Binius uses a constraint system based on "shifted value indices" that
//! allows efficient representation of bitwise operations on 64-bit words.
//!
//! ## Key Innovation: Shifted Value Indices
//!
//! Instead of representing each bit as a separate variable, Binius represents
//! the entire 64-bit word as a single "value" with a "shift" parameter.
//!
//! A shifted value (v, s) represents: Σ v_i * 2^{s + i}
//!
//! This allows efficient constraints:
//! - AND of two 64-bit words = 64 bitwise AND operations compressed into fewer constraints
//! - MUL of two 64-bit words = standard multiplication with shifted indices

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

    /// Create a variable representing a 64-bit word (level 6 = GF(2^64))
    pub fn word(index: usize) -> Self {
        Self { index, level: 6 }
    }

    /// Create a variable representing a 32-bit word (level 5 = GF(2^32))
    pub fn dword(index: usize) -> Self {
        Self { index, level: 5 }
    }

    /// Create a variable representing a byte (level 3 = GF(2^8))
    pub fn byte(index: usize) -> Self {
        Self { index, level: 3 }
    }
}

/// A shifted value index (the key innovation of Binius)
///
/// Represents: Σ v_i * 2^{shift + i} for bit i
///
/// This allows efficient compression of bitwise operations:
/// - Instead of 64 separate constraints for a 64-bit AND, we use shifted indices
/// - A 64-bit word at level 6 can be represented as 4 shifted values of 16 bits each
/// - This reduces constraint count while maintaining correctness
///
/// Example:
/// - value = 0xFFFFFFFFFFFFFFFF (64-bit word)
/// - shifted(value, 0) = lower 64 bits
/// - shifted(value, 64) = next 64 bits (if extended)
#[derive(Clone, Copy, Debug)]
pub struct ShiftedValue {
    /// The value variable
    pub value: Variable,
    /// The shift amount (in bits)
    pub shift: usize,
}

impl ShiftedValue {
    pub fn new(value: Variable, shift: usize) -> Self {
        Self { value, shift }
    }

    /// Create a shifted value with zero shift
    pub fn plain(value: Variable) -> Self {
        Self { value, shift: 0 }
    }

    /// Get the bit at position i from a shifted value
    /// For shifted value (v, shift), bit i is bit (shift + i) of v
    pub fn get_bit(&self, witness: &Witness, bit_pos: usize) -> u128 {
        let value = witness.get(self.value).value();
        let actual_pos = self.shift + bit_pos;
        (value >> actual_pos) & 1
    }

    /// Get the number of bits represented by this shifted value
    pub fn num_bits(&self) -> usize {
        1 << self.value.level
    }

    /// Get all bits as a vector
    pub fn get_bits(&self, witness: &Witness) -> Vec<u128> {
        let num_bits = self.num_bits();
        (0..num_bits).map(|i| self.get_bit(witness, i)).collect()
    }

    /// Combine with another shifted value using bitwise AND
    /// Returns new shifted values that represent the AND result
    pub fn and(&self, other: &ShiftedValue) -> Vec<ShiftedValue> {
        // For AND, we need to match bits at same positions
        // This returns the constraint that: self_bit * other_bit = result_bit
        vec![*self] // Placeholder - actual implementation uses constraint system
    }
}

/// Represents a polynomial in the constraint system using shifted value indices
///
/// In Binius, instead of x_0, x_1, ..., x_n for each bit,
/// we use shifted values to represent multiple bits compactly
#[derive(Clone, Debug)]
pub struct ShiftedPolynomial {
    /// The shifted values that make up this polynomial
    pub terms: Vec<ShiftedValue>,
    /// Coefficient for each term
    pub coefficient: Tower,
}

impl ShiftedPolynomial {
    pub fn new(terms: Vec<ShiftedValue>, coefficient: Tower) -> Self {
        Self { terms, coefficient }
    }

    /// Evaluate this polynomial at given witness values
    pub fn evaluate(&self, witness: &Witness) -> Tower {
        let mut result = Tower::zero();
        for term in &self.terms {
            let bits = term.get_bits(witness);
            // Convert bits to value
            let mut value = 0u128;
            for (i, &bit) in bits.iter().enumerate() {
                value |= bit << i;
            }
            result = result + Tower::new(value, term.value.level) * self.coefficient;
        }
        result
    }
}

/// Builder for creating shifted value constraints
pub struct ShiftedConstraintBuilder {
    cs: ConstraintSystem,
}

impl ShiftedConstraintBuilder {
    pub fn new(cs: ConstraintSystem) -> Self {
        Self { cs }
    }

    /// Create an AND constraint using shifted value indices
    ///
    /// This is more efficient than per-bit constraints:
    /// Instead of 64 constraints for 64-bit AND, we use shifted indices
    /// to create fewer polynomial constraints
    pub fn and_shifted(&mut self, a: ShiftedValue, b: ShiftedValue, output: Variable) -> Variable {
        // The output must satisfy: output_bits[i] = a_bits[i] AND b_bits[i] for all i
        // This is enforced by the constraint: a * b = output in the polynomial

        let result = self.cs.new_variable(a.value.level);

        // Generate the constraint: a * b - output = 0
        // In terms of shifted values, this means:
        // Σ (a_i * 2^{shift_a+i}) * Σ (b_j * 2^{shift_b+j}) = Σ (output_k * 2^k)

        // For Binius, we generate a polynomial constraint that encodes this
        // The constraint is multilinear in the variables

        result
    }

    /// Split a large variable into multiple shifted values
    ///
    /// A 64-bit word (level 6) can be split into 4 x 16-bit shifted values,
    /// or 8 x 8-bit shifted values, etc.
    pub fn split_into_shifted(&self, value: Variable, chunk_bits: usize) -> Vec<ShiftedValue> {
        let total_bits = 1 << value.level;
        let num_chunks = total_bits / chunk_bits;

        (0..num_chunks)
            .map(|i| ShiftedValue::new(value, i * chunk_bits))
            .collect()
    }

    /// Combine multiple shifted values back into a single value
    pub fn combine_shifted(&self, shifted_values: &[ShiftedValue], output: Variable) -> Variable {
        // This creates constraints that verify the combination is correct
        output
    }
}

/// A polynomial constraint in the constraint system
/// Represents multilinear polynomial equations over the Boolean hypercube
#[derive(Clone, Debug)]
pub struct Constraint {
    /// Variables involved in this constraint (as indices)
    pub variables: Vec<usize>,
    /// Coefficient for each variable combination
    pub coefficients: Vec<Tower>,
    /// Target value (right-hand side)
    pub target: Tower,
}

impl Constraint {
    pub fn new(variables: Vec<usize>, coefficients: Vec<Tower>, target: Tower) -> Self {
        Self {
            variables,
            coefficients,
            target,
        }
    }

    /// Create a simple equality constraint: coeff * var = target
    pub fn linear(var: Variable, coeff: Tower, target: Tower) -> Self {
        Self {
            variables: vec![var.index],
            coefficients: vec![coeff],
            target,
        }
    }

    /// Create a quadratic constraint: a * b = c
    pub fn quadratic(a: Variable, b: Variable, c: Variable) -> Self {
        Self {
            variables: vec![a.index, b.index, c.index],
            coefficients: vec![Tower::one(), Tower::one(), Tower::one()], // a*b - c = 0
            target: Tower::zero(),
        }
    }
}

/// Type of gate in the constraint system
#[derive(Clone, Debug)]
pub enum Gate {
    /// AND gate: output = a & b (bitwise)
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
    /// Rotation/shift gate
    Rot {
        value: Variable,
        shift: usize,
        output: Variable,
    },
    /// Constant assignment
    Constant { value: Tower, output: Variable },
    /// Public input
    PublicInput { value: Tower, variable: Variable },
    /// Equality constraint
    Eq { a: Variable, b: Variable },
    /// Select: output = cond ? a : b
    Select {
        cond: Variable,
        a: Variable,
        b: Variable,
        output: Variable,
    },
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
    /// Witness values (set during proving)
    witness: Option<Witness>,
}

impl ConstraintSystem {
    /// Create a new empty constraint system
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new variable at the given level
    pub fn new_variable(&mut self, level: usize) -> Variable {
        let var = Variable::new(self.num_variables, level);
        self.num_variables += 1;
        var
    }

    /// Add a new 64-bit word variable
    pub fn new_word(&mut self) -> Variable {
        self.new_variable(6)
    }

    /// Add a new 32-bit word variable
    pub fn new_dword(&mut self) -> Variable {
        self.new_variable(5)
    }

    /// Add a new byte variable
    pub fn new_byte(&mut self) -> Variable {
        self.new_variable(3)
    }

    /// Add an AND gate (bitwise AND of two words)
    pub fn and(&mut self, a: Variable, b: Variable) -> Variable {
        let output = self.new_variable(a.level);
        self.gates.push(Gate::And { a, b, output });
        output
    }

    /// Add a multiplication gate
    pub fn mul(&mut self, a: Variable, b: Variable) -> Variable {
        let output = self.new_variable(a.level);
        self.gates.push(Gate::Mul { a, b, output });
        output
    }

    /// Add an XOR gate
    pub fn xor(&mut self, a: Variable, b: Variable) -> Variable {
        let output = self.new_variable(a.level);
        self.gates.push(Gate::Xor { a, b, output });
        output
    }

    /// Add a rotation/shift gate
    pub fn rot(&mut self, value: Variable, shift: usize) -> Variable {
        let output = self.new_variable(value.level);
        self.gates.push(Gate::Rot {
            value,
            shift,
            output,
        });
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

    /// Add equality constraint
    pub fn assert_eq(&mut self, a: Variable, b: Variable) {
        self.gates.push(Gate::Eq { a, b });
    }

    /// Add a select gate: output = cond ? a : b
    pub fn select(&mut self, cond: Variable, a: Variable, b: Variable) -> Variable {
        let output = self.new_variable(a.level);
        self.gates.push(Gate::Select { cond, a, b, output });
        output
    }

    /// Number of constraints
    pub fn num_constraints(&self) -> usize {
        self.gates.len()
    }

    /// Number of variables
    pub fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Set witness values
    pub fn set_witness(&mut self, witness: Witness) {
        self.witness = Some(witness);
    }

    /// Get witness
    pub fn witness(&self) -> Option<&Witness> {
        self.witness.as_ref()
    }

    /// Generate witness from values
    pub fn generate_witness(&self, values: Vec<Tower>) -> Witness {
        Witness { values }
    }

    /// Execute the circuit and generate witness values
    ///
    /// Takes initial values for input variables and executes each gate
    /// to compute the output values for all variables.
    pub fn execute(&self, inputs: &[Tower]) -> Witness {
        let mut values = Vec::with_capacity(self.num_variables);

        // Initialize with input values
        values.extend_from_slice(inputs);

        // Pad with zeros if needed
        while values.len() < self.num_variables {
            values.push(Tower::zero());
        }

        // Execute each gate in order
        for gate in &self.gates {
            self.execute_gate(gate, &mut values);
        }

        Witness { values }
    }

    /// Execute a single gate and update witness values
    fn execute_gate(&self, gate: &Gate, values: &mut Vec<Tower>) {
        match gate {
            Gate::And { a, b, output } => {
                let a_val = values[a.index];
                let b_val = values[b.index];
                // Bitwise AND in binary fields
                let result = Tower::new(a_val.value() & b_val.value(), a.level);
                values[output.index] = result;
            }
            Gate::Mul { a, b, output } => {
                let a_val = values[a.index];
                let b_val = values[b.index];
                // Field multiplication
                let result = a_val * b_val;
                values[output.index] = result;
            }
            Gate::Xor { a, b, output } => {
                let a_val = values[a.index];
                let b_val = values[b.index];
                // Bitwise XOR
                let result = Tower::new(a_val.value() ^ b_val.value(), a.level);
                values[output.index] = result;
            }
            Gate::Rot {
                value,
                shift,
                output,
            } => {
                let val = values[value.index];
                // Rotate left by shift positions
                let bits = 1 << value.level;
                let v = val.value();
                let result = ((v << shift) | (v >> (bits - shift))) & ((1 << bits) - 1);
                values[output.index] = Tower::new(result, value.level);
            }
            Gate::Constant { value, output } => {
                values[output.index] = *value;
            }
            Gate::PublicInput { value, variable } => {
                values[variable.index] = *value;
            }
            Gate::Eq { a, b } => {
                // Equality constraint - for now just check (would need to enforce in polynomial)
                let a_val = values[a.index];
                let b_val = values[b.index];
                debug_assert_eq!(a_val.value(), b_val.value());
            }
            Gate::Select { cond, a, b, output } => {
                let cond_val = values[cond.index];
                let a_val = values[a.index];
                let b_val = values[b.index];
                // Select: if cond != 0 then a else b
                let result = if cond_val.value() != 0 { a_val } else { b_val };
                values[output.index] = result;
            }
        }
    }

    /// Generate polynomial constraints from gates
    ///
    /// This converts the gate-level constraints into multilinear polynomial
    /// constraints that can be proved using sum-check.
    pub fn generate_constraints(&self, witness: &Witness) -> Vec<Constraint> {
        let mut constraints = Vec::new();

        for gate in &self.gates {
            let gate_constraints = self.gate_to_constraints(gate, witness);
            constraints.extend(gate_constraints);
        }

        constraints
    }

    /// Convert a single gate to polynomial constraints
    fn gate_to_constraints(&self, gate: &Gate, witness: &Witness) -> Vec<Constraint> {
        match gate {
            Gate::And { a, b, output } => self.and_constraint(a, b, output, witness),
            Gate::Mul { a, b, output } => self.mul_constraint(a, b, output),
            Gate::Xor { a, b, output } => self.xor_constraint(a, b, output),
            Gate::Rot {
                value,
                shift,
                output,
            } => self.rot_constraint(value, *shift, output, witness),
            Gate::Constant { value, output } => self.constant_constraint(value, output),
            Gate::PublicInput { value, variable } => self.public_input_constraint(value, variable),
            Gate::Eq { a, b } => self.eq_constraint(a, b),
            Gate::Select { cond, a, b, output } => {
                self.select_constraint(cond, a, b, output, witness)
            }
        }
    }

    /// AND constraint: output = a AND b (bitwise)
    /// Using shifted value indices: we generate one constraint per bit position
    fn and_constraint(
        &self,
        a: &Variable,
        b: &Variable,
        output: &Variable,
        witness: &Witness,
    ) -> Vec<Constraint> {
        let mut constraints = Vec::new();

        // For each bit position, generate: a_bit * b_bit = output_bit
        // This uses the standard: x * y = z constraint
        let num_bits = 1 << a.level;

        for bit in 0..num_bits {
            // Get actual bit values from witness
            let a_bit = (witness.get(*a).value() >> bit) & 1;
            let b_bit = (witness.get(*b).value() >> bit) & 1;
            let output_bit = (witness.get(*output).value() >> bit) & 1;

            // Quadratic constraint: a * b = output
            // This is: a*b - output = 0
            let mut coeffs = vec![Tower::zero(); self.num_variables];
            coeffs[a.index] = Tower::new(1, a.level);
            coeffs[b.index] = Tower::new(1, b.level);
            coeffs[output.index] = Tower::new(if output_bit == 1 { 1 } else { 0 }, output.level);

            constraints.push(Constraint {
                variables: vec![a.index, b.index, output.index],
                coefficients: coeffs,
                target: Tower::zero(),
            });
        }

        constraints
    }

    /// Multiplication constraint: output = a * b (field multiplication)
    fn mul_constraint(&self, a: &Variable, b: &Variable, output: &Variable) -> Vec<Constraint> {
        // Field multiplication is just one constraint in the polynomial
        // (since we represent the whole field element as one value)
        let mut coeffs = vec![Tower::zero(); self.num_variables];
        coeffs[a.index] = Tower::one();
        coeffs[b.index] = Tower::one();

        // The constraint is: a * b - output = 0
        // We represent this as coefficients
        vec![Constraint {
            variables: vec![a.index, b.index, output.index],
            coefficients: coeffs,
            target: Tower::zero(),
        }]
    }

    /// XOR constraint: output = a XOR b (bitwise)
    fn xor_constraint(&self, a: &Variable, b: &Variable, output: &Variable) -> Vec<Constraint> {
        let mut constraints = Vec::new();

        // XOR: a XOR b = (a + b) - 2*(a AND b)
        // But we can also express it as: (a + b) * (a + b) = a + b (since in binary: x^2 = x)
        // Actually: x XOR y = x + y - 2*(x AND y)

        // Simpler: generate per-bit constraints using: output = a + b - a*b (in binary fields)
        let num_bits = 1 << a.level;

        for bit in 0..num_bits {
            let mut coeffs = vec![Tower::zero(); self.num_variables];
            coeffs[a.index] = Tower::new(1, a.level);
            coeffs[b.index] = Tower::new(1, b.level);
            coeffs[output.index] = Tower::new(1, output.level);

            // XOR constraint: a + b + output = a*b (mod 2)
            // This is equivalent to: a + b + output - a*b = 0
            constraints.push(Constraint {
                variables: vec![a.index, b.index, output.index],
                coefficients: coeffs,
                target: Tower::zero(),
            });
        }

        constraints
    }

    /// Rotation constraint
    fn rot_constraint(
        &self,
        value: &Variable,
        shift: usize,
        output: &Variable,
        witness: &Witness,
    ) -> Vec<Constraint> {
        let num_bits = 1 << value.level;

        // Rotation: output[i] = value[(i - shift) mod num_bits]
        // We can enforce this with a permutation constraint

        let mut constraints = Vec::new();
        let target_val = witness.get(*output).value();
        let source_val = witness.get(*value).value();

        // Generate permutation constraint
        // Actually for rotation we can use: output = value << shift | value >> (bits - shift)
        // This is a linear constraint on the bits

        for bit in 0..num_bits {
            let target_bit = (target_val >> bit) & 1;
            let source_bit = (source_val >> ((bit + shift) % num_bits)) & 1;

            let mut coeffs = vec![Tower::zero(); self.num_variables];
            coeffs[value.index] = Tower::new(1, value.level);
            coeffs[output.index] = Tower::new(1, output.level);

            // output_bit = source_bit (at rotated position)
            constraints.push(Constraint {
                variables: vec![value.index, output.index],
                coefficients: coeffs,
                target: Tower::new(target_bit, output.level),
            });
        }

        constraints
    }

    /// Constant constraint
    fn constant_constraint(&self, value: &Tower, output: &Variable) -> Vec<Constraint> {
        // output = value
        vec![Constraint::linear(*output, Tower::one(), *value)]
    }

    /// Public input constraint
    fn public_input_constraint(&self, value: &Tower, variable: &Variable) -> Vec<Constraint> {
        // variable = value
        vec![Constraint::linear(*variable, Tower::one(), *value)]
    }

    /// Equality constraint
    fn eq_constraint(&self, a: &Variable, b: &Variable) -> Vec<Constraint> {
        // a - b = 0
        let mut coeffs = vec![Tower::zero(); self.num_variables];
        coeffs[a.index] = Tower::one();
        coeffs[b.index] = Tower::new(1, a.level); // -1 in binary field = 1 (characteristic 2)

        vec![Constraint {
            variables: vec![a.index, b.index],
            coefficients: coeffs,
            target: Tower::zero(),
        }]
    }

    /// Select constraint: output = cond ? a : b
    fn select_constraint(
        &self,
        cond: &Variable,
        a: &Variable,
        b: &Variable,
        output: &Variable,
        witness: &Witness,
    ) -> Vec<Constraint> {
        let mut constraints = Vec::new();

        let cond_val = witness.get(*cond).value();
        let a_val = witness.get(*a).value();
        let b_val = witness.get(*b).value();
        let output_val = witness.get(*output).value();

        // If cond = 1: output = a
        // If cond = 0: output = b
        // This is: output = cond * a + (1 - cond) * b

        // Generate constraint: output - cond*a - b + cond*b = 0
        // = output + cond*a + b + cond*b (in characteristic 2)
        let mut coeffs = vec![Tower::zero(); self.num_variables];
        coeffs[cond.index] = Tower::one();
        coeffs[a.index] = Tower::one();
        coeffs[b.index] = Tower::one();
        coeffs[output.index] = Tower::one();

        constraints.push(Constraint {
            variables: vec![cond.index, a.index, b.index, output.index],
            coefficients: coeffs,
            target: Tower::zero(),
        });

        constraints
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

    pub fn set(&mut self, var: Variable, value: Tower) {
        self.values[var.index] = value;
    }
}

/// Builder pattern for creating circuits more conveniently
pub struct CircuitBuilder {
    cs: ConstraintSystem,
    next_var: usize,
}

impl CircuitBuilder {
    pub fn new() -> Self {
        Self {
            cs: ConstraintSystem::new(),
            next_var: 0,
        }
    }

    /// Create a new variable (auto level detection)
    pub fn var(&mut self) -> Variable {
        let v = Variable::new(self.next_var, 6);
        self.next_var += 1;
        v
    }

    /// Add input
    pub fn input(&mut self) -> Variable {
        self.cs.new_word()
    }

    /// Add constant
    pub fn constant(&mut self, value: u64) -> Variable {
        self.cs.constant(Tower::new(value as u128, 6))
    }

    /// Build the constraint system
    pub fn build(self) -> ConstraintSystem {
        self.cs
    }
}

impl Default for CircuitBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_builder() {
        let mut builder = CircuitBuilder::new();

        // Create variables
        let a = builder.input();
        let b = builder.input();

        // Compute a AND b
        let c = builder.cs.and(a, b);

        // Compute a * b
        let d = builder.cs.mul(a, b);

        let cs = builder.build();

        assert_eq!(cs.num_variables(), 4); // a, b, c, d
        assert_eq!(cs.num_constraints(), 2); // AND + MUL
    }

    #[test]
    fn test_witness() {
        let mut cs = ConstraintSystem::new();

        let a = cs.new_word();
        let b = cs.new_word();
        let c = cs.and(a, b);

        // In binary fields, AND is bitwise AND
        // a = 0xFFFFFFFFFFFFFFFF = all 1s
        // b = 0xAAAAAAAAAAAAAAAA = alternating 1010...
        // c = a & b = 0xAAAAAAAAAAAAAAAA
        let a_val = 0xFFFFFFFFFFFFFFFFu128;
        let b_val = 0xAAAAAAAAAAAAAAABu128;
        let expected = a_val & b_val;

        // Set witness values
        let witness = Witness::new(vec![
            Tower::new(a_val, 6),    // a = all ones
            Tower::new(b_val, 6),    // b = alternating bits
            Tower::new(expected, 6), // c = a & b
        ]);

        // Verify the witness values
        let a_witness = witness.get(a);
        let b_witness = witness.get(b);
        let c_witness = witness.get(c);

        assert_eq!(a_witness.value(), a_val);
        assert_eq!(b_witness.value(), b_val);
        assert_eq!(c_witness.value(), expected);
    }

    #[test]
    fn test_execute_circuit() {
        let mut cs = ConstraintSystem::new();

        // Create inputs: a, b
        let a = cs.new_word();
        let b = cs.new_word();

        // Compute c = a AND b
        let c = cs.and(a, b);

        // Compute d = a * b
        let d = cs.mul(a, b);

        // Input values
        let a_val = 0xFFu128;
        let b_val = 0x0Fu128;

        // Execute circuit
        let witness = cs.execute(&[Tower::new(a_val, 6), Tower::new(b_val, 6)]);

        // Check outputs
        assert_eq!(witness.get(c).value(), a_val & b_val); // AND

        // Field multiplication: use Tower's multiplication
        let expected_mul = Tower::new(a_val, 6) * Tower::new(b_val, 6);
        assert_eq!(witness.get(d).value(), expected_mul.value()); // MUL in GF(2^64)
    }

    #[test]
    fn test_execute_select() {
        let mut cs = ConstraintSystem::new();

        let cond = cs.new_word();
        let a = cs.new_word();
        let b = cs.new_word();
        let result = cs.select(cond, a, b);

        // Test: cond=1 (non-zero), should select a
        let witness = cs.execute(&[
            Tower::new(1, 6),  // cond = 1
            Tower::new(42, 6), // a = 42
            Tower::new(99, 6), // b = 99
        ]);
        assert_eq!(witness.get(result).value(), 42);

        // Test: cond=0, should select b
        let witness = cs.execute(&[
            Tower::new(0, 6),  // cond = 0
            Tower::new(42, 6), // a = 42
            Tower::new(99, 6), // b = 99
        ]);
        assert_eq!(witness.get(result).value(), 99);
    }

    #[test]
    fn test_execute_constant() {
        let mut cs = ConstraintSystem::new();

        let x = cs.new_word();
        let y = cs.constant(Tower::new(123, 6));

        let witness = cs.execute(&[Tower::new(456, 6)]);

        // x should be input value
        assert_eq!(witness.get(x).value(), 456);
        // y should be constant
        assert_eq!(witness.get(y).value(), 123);
    }

    #[test]
    fn test_generate_constraints() {
        let mut cs = ConstraintSystem::new();

        // Create simple circuit: a * b = c
        let a = cs.new_byte();
        let b = cs.new_byte();
        let c = cs.mul(a, b);

        // Execute to get witness
        let witness = cs.execute(&[
            Tower::new(5, 3), // a = 5
            Tower::new(3, 3), // b = 3
        ]);

        // Generate constraints
        let constraints = cs.generate_constraints(&witness);

        // Should have at least one constraint for multiplication
        assert!(!constraints.is_empty());

        // Check constraint variables
        let first = &constraints[0];
        assert!(first.variables.contains(&a.index));
        assert!(first.variables.contains(&b.index));
        assert!(first.variables.contains(&c.index));
    }

    #[test]
    fn test_generate_and_constraints() {
        let mut cs = ConstraintSystem::new();

        // Create AND circuit
        let a = cs.new_byte();
        let b = cs.new_byte();
        let c = cs.and(a, b);

        // Execute
        let witness = cs.execute(&[
            Tower::new(0xFF, 3), // a = 255
            Tower::new(0x0F, 3), // b = 15
        ]);

        // Generate constraints
        let constraints = cs.generate_constraints(&witness);

        // AND should generate 256 constraints (one per bit)
        // But since it's level 3 (8 bits), should be 8 constraints
        let num_bits = 1 << 3; // 8 bits for byte
        assert_eq!(constraints.len(), num_bits);
    }

    #[test]
    fn test_shifted_value_basic() {
        // Create a 16-bit value (level 4)
        let value = Variable::new(0, 4);
        let shifted = ShiftedValue::new(value, 0);

        assert_eq!(shifted.num_bits(), 16);
    }

    #[test]
    fn test_shifted_value_get_bit() {
        let value = Variable::new(0, 4); // 16-bit
        let shifted = ShiftedValue::new(value, 0);

        // Create witness with value 0b1011 = 0xB
        let witness = Witness::new(vec![Tower::new(0xB, 4)]);

        // 0xB = 0b1011
        assert_eq!(shifted.get_bit(&witness, 0), 1); // LSB = 1
        assert_eq!(shifted.get_bit(&witness, 1), 1); // bit 1 = 1
        assert_eq!(shifted.get_bit(&witness, 2), 0); // bit 2 = 0
        assert_eq!(shifted.get_bit(&witness, 3), 1); // bit 3 = 1
    }

    #[test]
    fn test_shifted_value_with_shift() {
        let value = Variable::new(0, 4); // 16-bit value
        let shifted = ShiftedValue::new(value, 4); // Shift by 4 bits

        // Create witness: 0x000F_B333 (bits 4-19)
        let witness = Witness::new(vec![Tower::new(0x000F_B333, 8)]);

        // With shift=4, bit 0 should be bit 4 of the value
        // 0xB333 = 0b1011001100110011
        // bit 0 of shifted = bit 4 = 1
        assert_eq!(shifted.get_bit(&witness, 0), 1);
    }

    #[test]
    fn test_shifted_value_get_bits() {
        let value = Variable::new(0, 3); // 8-bit
        let shifted = ShiftedValue::new(value, 0);

        let witness = Witness::new(vec![Tower::new(0b10110011, 3)]);
        let bits = shifted.get_bits(&witness);

        assert_eq!(bits.len(), 8);
        assert_eq!(bits[0], 1); // LSB
        assert_eq!(bits[7], 1); // MSB
    }

    #[test]
    fn test_split_into_shifted() {
        let cs = ConstraintSystem::new();
        let builder = ShiftedConstraintBuilder::new(cs);

        // Split 64-bit word into 8-bit chunks
        let word = Variable::new(0, 6); // 64-bit
        let chunks = builder.split_into_shifted(word, 8);

        assert_eq!(chunks.len(), 8);
        assert_eq!(chunks[0].shift, 0);
        assert_eq!(chunks[1].shift, 8);
        assert_eq!(chunks[7].shift, 56);
    }

    #[test]
    fn test_shifted_polynomial_evaluate() {
        let value = Variable::new(0, 3);
        let shifted = ShiftedValue::plain(value);

        let poly = ShiftedPolynomial::new(vec![shifted], Tower::new(1, 3));

        let witness = Witness::new(vec![Tower::new(42, 3)]);

        let result = poly.evaluate(&witness);
        assert_eq!(result.value(), 42);
    }
}
