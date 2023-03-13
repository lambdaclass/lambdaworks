use lambdaworks_math::field::{element::FieldElement, traits::IsField};

use crate::constraints::boundary::BoundaryConstraints;

pub struct Frame<F: IsField> {
    data: Vec<FieldElement<F>>,
    row_width: usize,
}

impl<F: IsField> Frame<F> {
    pub fn num_rows(&self) -> usize {
        self.data.len() / self.row_width
    }

    pub fn num_columns(&self) -> usize {
        self.row_width
    }

    pub fn get_row(&self, row_idx: usize) -> &[FieldElement<F>] {
        let row_offset = row_idx * self.row_width;
        &self.data[row_offset..row_offset + self.row_width]
    }

    pub fn get_row_mut(&mut self, row_idx: usize) -> &mut [FieldElement<F>] {
        let row_offset = row_idx * self.row_width;
        &mut self.data[row_offset..row_offset + self.row_width]
    }
}

pub trait AIR<F: IsField> {
    fn compute_transition(&self, frame: &Frame<F>) -> Vec<FieldElement<F>>;
    fn compute_boundary_constraints(&self) -> BoundaryConstraints<FieldElement<F>>;
}
