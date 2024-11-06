use crate::table::Table;
use itertools::Itertools;
use lambdaworks_math::{
    circle::{
        point::CirclePoint,
        polynomial::{evaluate_point, interpolate_cfft},
    },
    fft::errors::FFTError,
    field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field},
};
#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

/// A two-dimensional representation of an execution trace of the STARK
/// protocol.
///
/// For the moment it is mostly a wrapper around the `Table` struct. It is a
/// layer above the raw two-dimensional table, with functionality relevant to the
/// STARK protocol, such as the step size (number of consecutive rows of the table)
/// of the computation being proven.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct TraceTable {
    pub table: Table,
    pub num_columns: usize,
}

impl TraceTable {
    pub fn new(data: Vec<FieldElement<Mersenne31Field>>, num_columns: usize) -> Self {
        let table = Table::new(data, num_columns);
        Self { table, num_columns }
    }

    pub fn from_columns(columns: Vec<Vec<FieldElement<Mersenne31Field>>>) -> Self {
        println!("COLUMNS LEN: {}", columns.len());
        let num_columns = columns.len();
        let table = Table::from_columns(columns);
        Self { table, num_columns }
    }

    pub fn empty() -> Self {
        Self::new(Vec::new(), 0)
    }

    pub fn is_empty(&self) -> bool {
        self.table.width == 0
    }

    pub fn n_rows(&self) -> usize {
        self.table.height
    }

    pub fn n_cols(&self) -> usize {
        self.table.width
    }

    pub fn rows(&self) -> Vec<Vec<FieldElement<Mersenne31Field>>> {
        self.table.rows()
    }

    pub fn get_row(&self, row_idx: usize) -> &[FieldElement<Mersenne31Field>] {
        self.table.get_row(row_idx)
    }

    pub fn get_row_mut(&mut self, row_idx: usize) -> &mut [FieldElement<Mersenne31Field>] {
        self.table.get_row_mut(row_idx)
    }

    pub fn last_row(&self) -> &[FieldElement<Mersenne31Field>] {
        self.get_row(self.n_rows() - 1)
    }

    pub fn columns(&self) -> Vec<Vec<FieldElement<Mersenne31Field>>> {
        self.table.columns()
    }

    /// Given a slice of integer numbers representing column indexes, merge these columns into
    /// a one-dimensional vector.
    ///
    /// The particular way they are merged is not really important since this function is used to
    /// aggreagate values distributed across various columns with no importance on their ordering,
    /// such as to sort them.
    pub fn merge_columns(&self, column_indexes: &[usize]) -> Vec<FieldElement<Mersenne31Field>> {
        let mut data = Vec::with_capacity(self.n_rows() * column_indexes.len());
        for row_index in 0..self.n_rows() {
            for column in column_indexes {
                data.push(self.table.data[row_index * self.n_cols() + column].clone());
            }
        }
        data
    }

    pub fn compute_trace_polys(&self) -> Vec<Vec<FieldElement<Mersenne31Field>>> {
        let columns = self.columns();
        #[cfg(feature = "parallel")]
        let iter = columns.par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = columns.iter();
        // FIX: Replace the .to_vec()
        iter.map(|col| interpolate_cfft(col.to_vec()))
            .collect::<Vec<Vec<FieldElement<Mersenne31Field>>>>()
    }

    /// Given the padding length, appends the last row of the trace table
    /// that many times.
    /// This is useful for example when the desired trace length should be power
    /// of two, and only the last row is the one that can be appended without affecting
    /// the integrity of the constraints.
    pub fn pad_with_last_row(&mut self, padding_len: usize) {
        let last_row = self.last_row().to_vec();
        (0..padding_len).for_each(|_| {
            self.table.append_row(&last_row);
        })
    }

    /// Given a row index, a column index and a value, tries to set that location
    /// of the trace with the given value.
    /// The row_idx passed as argument may be greater than the max row index by 1. In this case,
    /// last row of the trace is cloned, and the value is set in that cloned row. Then, the row is
    /// appended to the end of the trace.
    pub fn set_or_extend(
        &mut self,
        row_idx: usize,
        col_idx: usize,
        value: &FieldElement<Mersenne31Field>,
    ) {
        debug_assert!(col_idx < self.n_cols());
        // NOTE: This is not very nice, but for how the function is being used at the moment,
        // the passed `row_idx` should never be greater than `self.n_rows() + 1`. This is just
        // an integrity check for ease in the developing process, we should think a better alternative
        // in the future.
        debug_assert!(row_idx <= self.n_rows() + 1);
        if row_idx >= self.n_rows() {
            let mut last_row = self.last_row().to_vec();
            last_row[col_idx] = value.clone();
            self.table.append_row(&last_row)
        } else {
            let row = self.get_row_mut(row_idx);
            row[col_idx] = value.clone();
        }
    }
}
pub struct LDETraceTable {
    pub(crate) table: Table,
    pub(crate) blowup_factor: usize,
}

impl LDETraceTable {
    pub fn new(
        data: Vec<FieldElement<Mersenne31Field>>,
        n_columns: usize,
        blowup_factor: usize,
    ) -> Self {
        let table = Table::new(data, n_columns);

        Self {
            table,
            blowup_factor,
        }
    }

    pub fn from_columns(
        columns: Vec<Vec<FieldElement<Mersenne31Field>>>,
        blowup_factor: usize,
    ) -> Self {
        let table = Table::from_columns(columns);

        Self {
            table,
            blowup_factor,
        }
    }

    pub fn num_cols(&self) -> usize {
        self.table.width
    }

    pub fn num_rows(&self) -> usize {
        self.table.height
    }

    pub fn get_row(&self, row_idx: usize) -> &[FieldElement<Mersenne31Field>] {
        self.table.get_row(row_idx)
    }

    pub fn get_table(&self, row: usize, col: usize) -> &FieldElement<Mersenne31Field> {
        self.table.get(row, col)
    }
}

/// Given a slice of trace polynomials, an evaluation point `x`, the frame offsets
/// corresponding to the computation of the transitions, and a primitive root,
/// outputs the trace evaluations of each trace polynomial over the values used to
/// compute a transition.
/// Example: For a simple Fibonacci computation, if t(x) is the trace polynomial of
/// the computation, this will output evaluations t(x), t(g * x), t(g^2 * z).
pub fn get_trace_evaluations(
    trace_polys: &[Vec<FieldElement<Mersenne31Field>>],
    point: &CirclePoint<Mersenne31Field>,
    frame_offsets: &[usize],
    group_generator: &CirclePoint<Mersenne31Field>,
) -> Table {
    let evaluation_points = frame_offsets
        .iter()
        .map(|offset| (group_generator * (*offset as u128)) + point)
        .collect_vec();

    let evaluations: Vec<_> = evaluation_points
        .iter()
        .flat_map(|eval_point| {
            trace_polys
                .iter()
                .map(|poly| evaluate_point(poly, eval_point))
        })
        .collect();

    let table_width = trace_polys.len();
    Table::new(evaluations, table_width)
}

pub fn columns2rows(
    columns: Vec<Vec<FieldElement<Mersenne31Field>>>,
) -> Vec<Vec<FieldElement<Mersenne31Field>>> {
    let num_rows = columns[0].len();
    let num_cols = columns.len();

    (0..num_rows)
        .map(|row_index| {
            (0..num_cols)
                .map(|col_index| columns[col_index][row_index].clone())
                .collect()
        })
        .collect()
}

// #[cfg(test)]
// mod test {
//     use super::TraceTable;
//     use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::F17};
//     type FE = FieldElement<F17>;

//     #[test]
//     fn test_cols() {
//         let col_1 = vec![FE::from(1), FE::from(2), FE::from(5), FE::from(13)];
//         let col_2 = vec![FE::from(1), FE::from(3), FE::from(8), FE::from(21)];

//         let trace_table = TraceTable::from_columns(vec![col_1.clone(), col_2.clone()]);
//         let res_cols = trace_table.columns();

//         assert_eq!(res_cols, vec![col_1, col_2]);
//     }
// }
