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
    pub inputs: [usize; 2],
}

impl Gate {
    pub fn new(gate_type: GateType, inputs: [usize; 2]) -> Self {
        Self { gate_type, inputs }
    }
}

/// A layer of gates in the circuit.
#[derive(Clone)]
pub struct CircuitLayer {
    pub layer: Vec<Gate>,
}

impl CircuitLayer {
    pub fn new(layer: Vec<Gate>) -> Self {
        Self { layer }
    }

    pub fn len(&self) -> usize {
        self.layer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layer.is_empty()
    }
}

/// An evaluation of a `Circuit` on some input.

pub struct CircuitEvaluation<F> {
    /// Evaluations on per-layer basis.
    pub layers: Vec<Vec<F>>,
}

impl<F: Copy> CircuitEvaluation<F> {
    /// Takes a gate label and outputs the corresponding gate's value at layer `layer`.
    pub fn w(&self, layer: usize, label: usize) -> F {
        self.layers[layer][label]
    }
}

#[derive(Debug, Clone)]
pub enum CircuitError {
    InputsNotPowerOfTwo,
    LayerNotPowerOfTwo(usize), // index of the layer that is not a power of two
}

/// The circuit in layered form.
#[derive(Clone)]
pub struct Circuit {
    /// First layer being the output layer, last layer being
    /// the input layer.
    layers: Vec<CircuitLayer>,

    /// Number of inputs
    num_inputs: usize,
    layer_num_vars: Vec<usize>, // log2 of number of gates per layer
    input_num_vars: usize,      // log2 of number of inputs
}

impl Circuit {
    pub fn new(layers: Vec<CircuitLayer>, num_inputs: usize) -> Result<Self, CircuitError> {
        if !num_inputs.is_power_of_two() {
            return Err(CircuitError::InputsNotPowerOfTwo);
        }
        let input_num_vars = num_inputs.trailing_zeros() as usize;
        let mut layer_num_vars = Vec::with_capacity(layers.len());
        for (i, layer) in layers.iter().enumerate() {
            if !layer.len().is_power_of_two() {
                return Err(CircuitError::LayerNotPowerOfTwo(i));
            }
            layer_num_vars.push(layer.len().trailing_zeros() as usize);
        }
        Ok(Self {
            layers,
            num_inputs,
            layer_num_vars,
            input_num_vars,
        })
    }

    pub fn num_vars_at(&self, layer: usize) -> Option<usize> {
        let num_gates = if let Some(layer) = self.layers.get(layer) {
            layer.len()
        } else if layer == self.layers.len() {
            self.num_inputs
        } else {
            return None;
        };

        Some((num_gates as u64).trailing_zeros() as usize)
    }

    /// Evaluate a `Circuit` on a given input.
    pub fn evaluate<F>(&self, input: &[FieldElement<F>]) -> CircuitEvaluation<FieldElement<F>>
    where
        F: IsField,
    {
        let mut layers = vec![];
        let mut current_input = input.to_vec();

        layers.push(current_input.clone());

        for layer in self.layers.iter().rev() {
            let temp_layer: Vec<_> = layer
                .layer
                .iter()
                .map(|e| match e.gate_type {
                    GateType::Add => {
                        current_input[e.inputs[0]].clone() + current_input[e.inputs[1]].clone()
                    }
                    GateType::Mul => {
                        current_input[e.inputs[0]].clone() * current_input[e.inputs[1]].clone()
                    }
                })
                .collect();

            layers.push(temp_layer.clone());
            current_input = temp_layer;
        }

        layers.reverse();
        CircuitEvaluation { layers }
    }

    /// The $\text{add}_i(a, b, c)$ polynomial value at layer $i$.
    pub fn add_i(&self, i: usize, a: usize, b: usize, c: usize) -> bool {
        let gate = &self.layers[i].layer[a];
        gate.gate_type == GateType::Add && gate.inputs[0] == b && gate.inputs[1] == c
    }

    /// The $\text{mul}_i(a, b, c)$ polynomial value at layer $i$.
    pub fn mul_i(&self, i: usize, a: usize, b: usize, c: usize) -> bool {
        let gate = &self.layers[i].layer[a];
        gate.gate_type == GateType::Mul && gate.inputs[0] == b && gate.inputs[1] == c
    }

    pub fn layers(&self) -> &[CircuitLayer] {
        &self.layers
    }

    pub fn num_outputs(&self) -> usize {
        self.layers[0].layer.len()
    }

    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    pub fn add_i_ext<F: IsField>(
        &self,
        r_i: &[FieldElement<F>],
        i: usize,
    ) -> DenseMultilinearPolynomial<F>
    where
        F::BaseType: Send + Sync + Copy,
    {
        let num_vars_current = self.layer_num_vars[i];
        let num_vars_next = if let Some(n) = self.layer_num_vars.get(i + 1) {
            *n
        } else {
            self.input_num_vars
        };
        let total_vars = num_vars_current + 2 * num_vars_next;
        let mut add_i_evals = vec![FieldElement::zero(); 1 << total_vars];
        for (a, gate) in self.layers[i].layer.iter().enumerate() {
            if gate.gate_type == GateType::Add {
                let b = gate.inputs[0];
                let c = gate.inputs[1];
                let idx = (a << (2 * num_vars_next)) | (b << num_vars_next) | c;
                add_i_evals[idx] = FieldElement::one();
            }
        }
        let mut p = DenseMultilinearPolynomial::new(add_i_evals);
        for val in r_i.iter() {
            p = p.fix_last_variable(val);
        }
        p
    }

    pub fn mul_i_ext<F: IsField>(
        &self,
        r_i: &[FieldElement<F>],
        i: usize,
    ) -> DenseMultilinearPolynomial<F>
    where
        F::BaseType: Send + Sync + Copy,
    {
        let num_vars_current = self.layer_num_vars[i];
        let num_vars_next = if let Some(n) = self.layer_num_vars.get(i + 1) {
            *n
        } else {
            self.input_num_vars
        };
        let total_vars = num_vars_current + 2 * num_vars_next;
        let mut mul_i_evals = vec![FieldElement::zero(); 1 << total_vars];
        for (a, gate) in self.layers[i].layer.iter().enumerate() {
            if gate.gate_type == GateType::Mul {
                let b = gate.inputs[0];
                let c = gate.inputs[1];
                let idx = (a << (2 * num_vars_next)) | (b << num_vars_next) | c;
                mul_i_evals[idx] = FieldElement::one();
            }
        }
        let mut p = DenseMultilinearPolynomial::new(mul_i_evals);
        for val in r_i.iter() {
            p = p.fix_last_variable(val);
        }
        p
    }
}
