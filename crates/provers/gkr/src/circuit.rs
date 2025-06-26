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
    pub ttype: GateType,

    /// Two inputs, indexes into the previous layer gates outputs.
    pub inputs: [usize; 2],
}

impl Gate {
    /// Create a new `Gate`.
    pub fn new(ttype: GateType, inputs: [usize; 2]) -> Self {
        Self { ttype, inputs }
    }
}

/// A layer of gates in the circuit.
#[derive(Clone)]
pub struct CircuitLayer {
    pub layer: Vec<Gate>,
}

impl CircuitLayer {
    /// Create a new `CircuitLayer`.
    pub fn new(layer: Vec<Gate>) -> Self {
        Self { layer }
    }

    /// The length of the layer.
    pub fn len(&self) -> usize {
        self.layer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layer.is_empty()
    }
}

/// An evaluation of a `Circuit` on some input.
/// Stores every circuit layer interediary evaluations and the
/// circuit evaluation outputs.
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

/// The circuit in layered form.
#[derive(Clone)]
pub struct Circuit {
    /// First layer being the output layer, last layer being
    /// the input layer.
    layers: Vec<CircuitLayer>,

    /// Number of inputs
    num_inputs: usize,
}

impl Circuit {
    pub fn new(layers: Vec<CircuitLayer>, num_inputs: usize) -> Self {
        Self { layers, num_inputs }
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
                .map(|e| match e.ttype {
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

        gate.ttype == GateType::Add && gate.inputs[0] == b && gate.inputs[1] == c
    }

    /// The $\text{mul}_i(a, b, c)$ polynomial value at layer $i$.
    pub fn mul_i(&self, i: usize, a: usize, b: usize, c: usize) -> bool {
        let gate = &self.layers[i].layer[a];

        gate.ttype == GateType::Mul && gate.inputs[0] == b && gate.inputs[1] == c
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
        let mut add_i_evals: Vec<FieldElement<F>> = vec![];
        let num_vars_current = (self.layers[i].len() as f64).log2() as usize;

        let num_vars_next = (self
            .layers
            .get(i + 1)
            .map(|c| c.len())
            .unwrap_or(self.num_inputs) as f64)
            .log2() as usize;

        for c in 0..1 << num_vars_next {
            for b in 0..1 << num_vars_next {
                for a in 0..1 << num_vars_current {
                    add_i_evals.push(if self.add_i(i, a, b, c) {
                        FieldElement::one()
                    } else {
                        FieldElement::zero()
                    });
                }
            }
        }

        let add_i = DenseMultilinearPolynomial::new(add_i_evals);
        let mut p = add_i;
        for (_i, val) in r_i.iter().enumerate() {
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
        let mut mul_i_evals: Vec<FieldElement<F>> = vec![];
        let num_vars_current = (self.layers[i].len() as f64).log2() as usize;

        let num_vars_next = (self
            .layers
            .get(i + 1)
            .map(|c| c.len())
            .unwrap_or(self.num_inputs) as f64)
            .log2() as usize;

        for c in 0..1 << num_vars_next {
            for b in 0..1 << num_vars_next {
                for a in 0..1 << num_vars_current {
                    mul_i_evals.push(if self.mul_i(i, a, b, c) {
                        FieldElement::one()
                    } else {
                        FieldElement::zero()
                    });
                }
            }
        }

        let mul_i = DenseMultilinearPolynomial::new(mul_i_evals);
        let mut p = mul_i;
        for (_i, val) in r_i.iter().enumerate() {
            p = p.fix_last_variable(val);
        }
        p
    }
}
