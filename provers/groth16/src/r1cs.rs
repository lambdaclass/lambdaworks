use crate::common::FrElement;

/// R1CS representation of an Arithmetic Program
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constraint {
    pub a: Vec<FrElement>,
    pub b: Vec<FrElement>,
    pub c: Vec<FrElement>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1CS {
    pub constraints: Vec<Constraint>,
    pub number_of_inputs: usize,
    pub number_of_outputs: usize,
}

impl R1CS {
    pub fn new(
        constraints: Vec<Constraint>,
        number_of_inputs: usize,
        number_of_outputs: usize,
    ) -> Self {
        let all_same_length = constraints
            .iter()
            .all(|v| v.a.len() == constraints[0].a.len());
        assert!(all_same_length);

        Self {
            constraints,
            number_of_inputs,
            number_of_outputs,
        }
    }

    pub fn new_with_matrixes(
        a: Vec<Vec<FrElement>>,
        b: Vec<Vec<FrElement>>,
        c: Vec<Vec<FrElement>>,
        num_inputs: usize,
        num_outputs: usize,
    ) -> Self {
        let mut constraints: Vec<Constraint> = Vec::with_capacity(a.len());
        // TO DO:
        // - Check if sizes match
        // - Remove clones
        for i in 0..a.len() {
            constraints.push(Constraint {
                a: a[i].clone(),
                b: b[i].clone(),
                c: c[i].clone(),
            })
        }
        R1CS::new(constraints, num_inputs, num_outputs)
    }

    pub fn number_of_constraints(&self) -> usize {
        self.constraints.len()
    }

    pub fn witness_size(&self) -> usize {
        self.constraints[0].a.len()
    }
}

impl Constraint {
    #[allow(dead_code)]
    pub fn verify_solution(self, s: &[FrElement]) -> bool {
        inner_product(&self.a, s) * inner_product(&self.b, s) == inner_product(&self.c, s)
    }
}

pub fn inner_product(v1: &[FrElement], v2: &[FrElement]) -> FrElement {
    v1.iter()
        .zip(v2)
        .map(|(x, y)| x * y)
        .fold(FrElement::zero(), |x, y| x + y)
}
