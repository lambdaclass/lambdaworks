use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;

/// A type of a gate in the Circuit.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum GateType {
    /// An addition gate.
    Add,
    /// A multiplication gate.
    Mul,
}

/// A gate in the Circuit.
#[derive(Clone, Copy)]
pub struct Gate {
    /// A type of the gate.
    pub gate_type: GateType,

    /// Two inputs, indexes into the previous layer gates outputs.
    pub inputs_idx: [usize; 2],
}

impl Gate {
    pub fn new(gate_type: GateType, inputs_idx: [usize; 2]) -> Self {
        Self {
            gate_type,
            inputs_idx,
        }
    }
}

/// A layer of gates in the circuit.
#[derive(Clone)]
pub struct CircuitLayer {
    pub gates: Vec<Gate>,
    pub num_of_vars: usize, // log2 of number of gates in this layer
}

impl CircuitLayer {
    pub fn new(gates: Vec<Gate>) -> Self {
        let num_of_vars = if gates.is_empty() {
            0
        } else {
            gates.len().next_power_of_two().trailing_zeros() as usize
        };
        Self { gates, num_of_vars }
    }

    pub fn len(&self) -> usize {
        self.gates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }
}

#[derive(Debug, Clone)]
pub enum CircuitError {
    InputsNotPowerOfTwo,
    LayerNotPowerOfTwo(usize),
    GateInputsError(usize),
    EmptyCircuitError,
}

/// The circuit in layered form.
///
/// ## Circuit Structure
///
/// Circuits are organized in layers, with each layer containing gates that operate on outputs from the previous layer:
///
/// ```text
/// Output:     o          o     <- circuit.layers[0]
///           /   \      /   \
///          o     o    o     o  <- circuit.layers[1]
///                  ...        
///            o   o   o   o     <- circuit.layers[layer.len() - 1]
///           / \ / \ / \ / \
/// Input:    o o o o o o o o
/// ```
///
/// - The top nodes are the circuit outputs (layers[0]).
/// - The bottom nodes are the circuit inputs, they don't belong to the vector `layers`.
/// - Each layer contains gates; edges represent wiring between layers.
/// - The circuit is evaluated from inputs upward, but layers are stored from output to input.
#[derive(Clone)]
pub struct Circuit {
    /// First layer is the output layer. It doesn't include the input layer.
    layers: Vec<CircuitLayer>,

    /// Number of inputs
    num_inputs: usize,
    input_num_vars: usize, // log2 of number of inputs
}

/// An evaluation of a `Circuit` on some input.
/// Stores the outputs, every circuit layer intermediate evaluations and the inputs
pub struct CircuitEvaluation<F: IsField> {
    /// Evaluations on per-layer. First layer is the output and last layer is the input.
    pub layers: Vec<Vec<FieldElement<F>>>,
}

impl Circuit {
    pub fn new(layers: Vec<CircuitLayer>, num_inputs: usize) -> Result<Self, CircuitError> {
        if layers.is_empty() {
            return Err(CircuitError::EmptyCircuitError);
        }

        if !num_inputs.is_power_of_two() {
            return Err(CircuitError::InputsNotPowerOfTwo);
        }

        let input_num_vars = num_inputs.trailing_zeros() as usize;

        // Validate that each layer has power-of-two gates
        for (i, layer) in layers.iter().enumerate() {
            if !layer.len().is_power_of_two() {
                return Err(CircuitError::LayerNotPowerOfTwo(i));
            }
        }

        // Validate that gate inputs in each layer don't exceed the next layer number of gates
        for (i, layer_pair) in layers.windows(2).enumerate() {
            let current_layer = &layer_pair[0];
            let next_layer = &layer_pair[1];
            let next_layer_gates = next_layer.len();

            if current_layer.gates.iter().any(|gate| {
                let [a, b] = gate.inputs_idx;
                a >= next_layer_gates || b >= next_layer_gates
            }) {
                return Err(CircuitError::GateInputsError(i));
            }
        }

        // Validate that the last layer gate inputs don't exceed the number of inputs
        if let Some(last_layer) = layers.last() {
            if last_layer.gates.iter().any(|gate| {
                let [a, b] = gate.inputs_idx;
                a >= num_inputs || b >= num_inputs
            }) {
                return Err(CircuitError::GateInputsError(layers.len() - 1));
            }
        }

        Ok(Self {
            layers,
            num_inputs,
            input_num_vars,
        })
    }

    pub fn num_vars_at(&self, layer: usize) -> Option<usize> {
        if let Some(layer) = self.layers.get(layer) {
            Some(layer.num_of_vars)
        } else if layer == self.layers.len() {
            Some(self.input_num_vars)
        } else {
            None
        }
    }

    /// Evaluate a `Circuit` on a given input.
    pub fn evaluate<F>(&self, input: &[FieldElement<F>]) -> CircuitEvaluation<F>
    where
        F: IsField,
    {
        let mut layers = Vec::with_capacity(self.layers.len() + 1);
        let mut current_input = input.to_vec();

        layers.push(current_input.clone());

        for layer in self.layers.iter().rev() {
            let temp_layer: Vec<_> = layer
                .gates
                .iter()
                .map(|gate| match gate.gate_type {
                    GateType::Add => {
                        &current_input[gate.inputs_idx[0]] + &current_input[gate.inputs_idx[1]]
                    }
                    GateType::Mul => {
                        &current_input[gate.inputs_idx[0]] * &current_input[gate.inputs_idx[1]]
                    }
                })
                .collect();

            layers.push(temp_layer.clone());
            current_input = temp_layer;
        }

        // Reverse the order so that the first layer is the output layer.
        layers.reverse();
        CircuitEvaluation { layers }
    }

    /// The add_i(a, b, c) polynomial value at layer i.
    pub fn add_i(&self, i: usize, a: usize, b: usize, c: usize) -> bool {
        let gate = &self.layers[i].gates[a];
        gate.gate_type == GateType::Add && gate.inputs_idx[0] == b && gate.inputs_idx[1] == c
    }

    /// The mul_i(a, b, c) polynomial value at layer i.
    pub fn mul_i(&self, i: usize, a: usize, b: usize, c: usize) -> bool {
        let gate = &self.layers[i].gates[a];
        gate.gate_type == GateType::Mul && gate.inputs_idx[0] == b && gate.inputs_idx[1] == c
    }

    pub fn layers(&self) -> &[CircuitLayer] {
        &self.layers
    }

    pub fn num_outputs(&self) -> usize {
        self.layers[0].gates.len()
    }

    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// The multilinear polynomial extension of the function `add_i(a, b, c)`, where `a` is fixed at `r_i`.
    pub fn add_i_ext<F: IsField>(
        &self,
        r_i: &[FieldElement<F>],
        i: usize,
    ) -> DenseMultilinearPolynomial<F>
    where
        F::BaseType: Send + Sync + Copy,
    {
        let num_vars_current = self.layers[i].num_of_vars;
        let num_vars_next = if let Some(layer) = self.layers.get(i + 1) {
            layer.num_of_vars
        } else {
            self.input_num_vars
        };
        let total_vars = num_vars_current + 2 * num_vars_next;
        let mut add_i_evals = vec![FieldElement::zero(); 1 << total_vars];

        // For each Add gate, we set the corresponding index `a || b || c` in the evaluation vector to one.
        for (a, gate) in self.layers[i].gates.iter().enumerate() {
            if gate.gate_type == GateType::Add {
                let b = gate.inputs_idx[0];
                let c = gate.inputs_idx[1];
                let idx = (a << (2 * num_vars_next)) | (b << num_vars_next) | c;
                add_i_evals[idx] = FieldElement::one();
            }
        }

        let mut add_i_poly = DenseMultilinearPolynomial::new(add_i_evals);
        for val in r_i.iter() {
            add_i_poly = add_i_poly.fix_first_variable(val);
        }
        add_i_poly
    }

    /// The multilinear polynomial extension of the function `mul_i(a, b, c)`, where `a` is fixed at `r_i`.
    pub fn mul_i_ext<F: IsField>(
        &self,
        r_i: &[FieldElement<F>],
        i: usize,
    ) -> DenseMultilinearPolynomial<F>
    where
        F::BaseType: Send + Sync + Copy,
    {
        let num_vars_current = self.layers[i].num_of_vars;
        let num_vars_next = if let Some(layer) = self.layers.get(i + 1) {
            layer.num_of_vars
        } else {
            self.input_num_vars
        };
        let total_vars = num_vars_current + 2 * num_vars_next;
        let mut mul_i_evals = vec![FieldElement::zero(); 1 << total_vars];

        // For each Mul gate, we set the corresponding index `a || b || c` in the evaluation vector to one.
        for (a, gate) in self.layers[i].gates.iter().enumerate() {
            if gate.gate_type == GateType::Mul {
                let b = gate.inputs_idx[0];
                let c = gate.inputs_idx[1];
                let idx = (a << (2 * num_vars_next)) | (b << num_vars_next) | c;
                mul_i_evals[idx] = FieldElement::one();
            }
        }

        let mut mul_i_poly = DenseMultilinearPolynomial::new(mul_i_evals);
        for val in r_i.iter() {
            mul_i_poly = mul_i_poly.fix_first_variable(val);
        }
        mul_i_poly
    }
}
