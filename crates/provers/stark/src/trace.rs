use crate::table::Table;
use itertools::Itertools;
use lambdaworks_math::fft::errors::FFTError;
use lambdaworks_math::field::traits::{IsField, IsSubFieldOf};
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
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
pub struct TraceTable<F, E>
where
    E: IsField,
    F: IsSubFieldOf<E> + IsField,
{
    pub main_table: Table<F>,
    pub aux_table: Table<E>,
    pub num_main_columns: usize,
    pub num_aux_columns: usize,
    pub step_size: usize,
}

impl<F, E> TraceTable<F, E>
where
    E: IsField,
    F: IsSubFieldOf<E> + IsFFTField,
{
    pub fn new(
        main_data: Vec<FieldElement<F>>,
        aux_data: Vec<FieldElement<E>>,
        num_main_columns: usize,
        num_aux_columns: usize,
        step_size: usize,
    ) -> Self {
        let main_table = Table::new(main_data, num_main_columns);
        let aux_table = Table::new(aux_data, num_aux_columns);

        Self {
            main_table,
            aux_table,
            num_main_columns,
            num_aux_columns,
            step_size,
        }
    }

    /// Creates a new TraceTable from from a one-dimensional array in row major order and the intended width of the table.
    /// Step size is how many are needed to represent a state of the VM
    pub fn new_main(
        main_data: Vec<FieldElement<F>>,
        num_main_columns: usize,
        step_size: usize,
    ) -> Self {
        let num_aux_columns = 0;
        let main_table = Table::new(main_data, num_main_columns);
        let aux_table = Table::new(Vec::new(), num_aux_columns);

        Self {
            main_table,
            aux_table,
            num_main_columns,
            num_aux_columns,
            step_size,
        }
    }

    /// Creates a new TraceTable from its colummns
    /// Step size is how many are needed to represent a state of the VM
    pub fn from_columns(
        main_columns: Vec<Vec<FieldElement<F>>>,
        aux_columns: Vec<Vec<FieldElement<E>>>,
        step_size: usize,
    ) -> Self {
        let num_main_columns = main_columns.len();
        let num_aux_columns = aux_columns.len();

        let main_table = Table::from_columns(main_columns);
        let aux_table = Table::from_columns(aux_columns);

        Self {
            main_table,
            aux_table,
            num_main_columns,
            num_aux_columns,
            step_size,
        }
    }

    pub fn from_columns_main(columns: Vec<Vec<FieldElement<F>>>, step_size: usize) -> Self {
        let num_main_columns = columns.len();
        let num_aux_columns = 0;
        let main_table = Table::from_columns(columns);
        let aux_table = Table::from_columns(Vec::new());

        Self {
            main_table,
            aux_table,
            num_main_columns,
            num_aux_columns,
            step_size,
        }
    }

    pub fn empty() -> Self {
        Self::new(Vec::new(), Vec::new(), 0, 0, 0)
    }

    pub fn is_empty(&self) -> bool {
        self.main_table.width == 0 && self.aux_table.width == 0
    }

    pub fn num_rows(&self) -> usize {
        self.main_table.height
    }

    pub fn num_steps(&self) -> usize {
        debug_assert!((self.main_table.height % self.step_size) == 0);
        self.main_table.height / self.step_size
    }

    /// Given a particular step of the computation represented on the trace,
    /// returns the row of the underlying table.
    pub fn step_to_row(&self, step: usize) -> usize {
        self.step_size * step
    }

    pub fn num_cols(&self) -> usize {
        self.main_table.width + self.aux_table.width
    }

    pub fn columns_main(&self) -> Vec<Vec<FieldElement<F>>> {
        self.main_table.columns()
    }

    pub fn columns_aux(&self) -> Vec<Vec<FieldElement<E>>> {
        self.aux_table.columns()
    }

    /// Given a row and a column index, gives stored value in that position
    pub fn get_main(&self, row: usize, col: usize) -> &FieldElement<F> {
        self.main_table.get(row, col)
    }

    /// Given a row and a column index, gives stored value in that position
    pub fn get_aux(&self, row: usize, col: usize) -> &FieldElement<E> {
        self.aux_table.get(row, col)
    }

    pub fn set_main(&mut self, row: usize, col: usize, value: FieldElement<F>) {
        self.main_table.set(row, col, value);
    }

    pub fn set_aux(&mut self, row: usize, col: usize, value: FieldElement<E>) {
        self.aux_table.set(row, col, value);
    }

    /// Allocates an auxiliary trace table with zero-initialized columns.
    /// Used by AirWithLogUp to dynamically add aux columns during build_auxiliary_trace.
    pub fn allocate_aux_table(&mut self, num_aux_columns: usize) {
        let num_rows = self.num_rows();
        let aux_data = vec![FieldElement::<E>::zero(); num_rows * num_aux_columns];
        self.aux_table = Table::new(aux_data, num_aux_columns);
        self.num_aux_columns = num_aux_columns;
    }

    pub fn allocate_with_zeros(
        num_steps: usize,
        num_main_columns: usize,
        num_aux_columns: usize,
        step_size: usize,
    ) -> TraceTable<F, E> {
        let main_data = vec![FieldElement::<F>::zero(); step_size * num_steps * num_main_columns];
        let aux_data = vec![FieldElement::<E>::zero(); step_size * num_steps * num_aux_columns];
        TraceTable::new(
            main_data,
            aux_data,
            num_main_columns,
            num_aux_columns,
            step_size,
        )
    }

    pub fn compute_trace_polys_main<S>(&self) -> Result<Vec<Polynomial<FieldElement<F>>>, FFTError>
    where
        S: IsFFTField + IsSubFieldOf<F>,
        FieldElement<F>: Send + Sync,
    {
        let columns = self.columns_main();
        #[cfg(feature = "parallel")]
        let iter = columns.par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = columns.iter();

        iter.map(|col| Polynomial::interpolate_fft::<S>(col))
            .collect::<Result<Vec<Polynomial<FieldElement<F>>>, FFTError>>()
    }

    pub fn compute_trace_polys_aux<S>(&self) -> Result<Vec<Polynomial<FieldElement<E>>>, FFTError>
    where
        S: IsFFTField + IsSubFieldOf<F>,
        FieldElement<E>: Send + Sync,
    {
        let columns = self.columns_aux();
        #[cfg(feature = "parallel")]
        let iter = columns.par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = columns.iter();

        iter.map(|col| Polynomial::interpolate_fft::<F>(col))
            .collect::<Result<Vec<Polynomial<FieldElement<E>>>, FFTError>>()
    }

    pub fn get_column_main(&self, col_idx: usize) -> Vec<FieldElement<F>> {
        self.main_table.get_column(col_idx)
    }

    pub fn get_column_aux(&self, col_idx: usize) -> Vec<FieldElement<E>> {
        self.aux_table.get_column(col_idx)
    }
}
pub struct LDETraceTable<F, E>
where
    E: IsField,
    F: IsSubFieldOf<E> + IsField,
{
    pub(crate) main_table: Table<F>,
    pub(crate) aux_table: Table<E>,
    pub(crate) lde_step_size: usize,
    pub(crate) blowup_factor: usize,
}

impl<F, E> LDETraceTable<F, E>
where
    E: IsField,
    F: IsSubFieldOf<E>,
{
    pub fn new(
        main_data: Vec<FieldElement<F>>,
        aux_data: Vec<FieldElement<E>>,
        n_columns: usize,
        trace_step_size: usize,
        blowup_factor: usize,
    ) -> Self {
        let main_table = Table::new(main_data, n_columns);
        let aux_table = Table::new(aux_data, n_columns);
        let lde_step_size = trace_step_size * blowup_factor;

        Self {
            main_table,
            aux_table,
            lde_step_size,
            blowup_factor,
        }
    }

    pub fn from_columns(
        main_columns: Vec<Vec<FieldElement<F>>>,
        aux_columns: Vec<Vec<FieldElement<E>>>,
        trace_step_size: usize,
        blowup_factor: usize,
    ) -> Self {
        Self::from_columns_ref(&main_columns, &aux_columns, trace_step_size, blowup_factor)
    }

    /// Creates an LDETraceTable from borrowed column slices, avoiding cloning the input data.
    pub fn from_columns_ref(
        main_columns: &[Vec<FieldElement<F>>],
        aux_columns: &[Vec<FieldElement<E>>],
        trace_step_size: usize,
        blowup_factor: usize,
    ) -> Self {
        let main_table = Table::from_columns_ref(main_columns);
        let aux_table = Table::from_columns_ref(aux_columns);
        let lde_step_size = trace_step_size * blowup_factor;

        Self {
            main_table,
            aux_table,
            lde_step_size,
            blowup_factor,
        }
    }

    pub fn num_cols(&self) -> usize {
        self.main_table.width + self.aux_table.width
    }

    pub fn num_rows(&self) -> usize {
        self.main_table.height
    }

    pub fn get_main_row(&self, row_idx: usize) -> &[FieldElement<F>] {
        self.main_table.get_row(row_idx)
    }

    pub fn get_aux_row(&self, row_idx: usize) -> &[FieldElement<E>] {
        self.aux_table.get_row(row_idx)
    }

    pub fn get_main(&self, row: usize, col: usize) -> &FieldElement<F> {
        self.main_table.get(row, col)
    }

    pub fn get_aux(&self, row: usize, col: usize) -> &FieldElement<E> {
        self.aux_table.get(row, col)
    }

    pub fn num_steps(&self) -> usize {
        debug_assert!((self.main_table.height % self.lde_step_size) == 0);
        self.main_table.height / self.lde_step_size
    }

    pub fn step_to_row(&self, step: usize) -> usize {
        self.lde_step_size * step
    }
}

/// Given a slice of trace polynomials, an evaluation point `x`, the frame offsets
/// corresponding to the computation of the transitions, and a primitive root,
/// outputs the trace evaluations of each trace polynomial over the values used to
/// compute a transition.
/// Example: For a simple Fibonacci computation, if t(x) is the trace polynomial of
/// the computation, this will output evaluations t(x), t(g * x), t(g^2 * z).
pub fn get_trace_evaluations<F, E>(
    main_trace_polys: &[Polynomial<FieldElement<F>>],
    aux_trace_polys: &[Polynomial<FieldElement<E>>],
    x: &FieldElement<E>,
    frame_offsets: &[usize],
    primitive_root: &FieldElement<F>,
    step_size: usize,
) -> Table<E>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    let evaluation_points = frame_offsets
        .iter()
        .flat_map(|offset| {
            let exponents_range_start = offset * step_size;
            let exponents_range_end = (offset + 1) * step_size;
            (exponents_range_start..exponents_range_end).collect_vec()
        })
        .map(|exponent| primitive_root.pow(exponent) * x)
        .collect_vec();

    let main_evaluations = evaluation_points
        .iter()
        .map(|eval_point| {
            main_trace_polys
                .iter()
                .map(|main_poly| main_poly.evaluate(eval_point))
                .collect_vec()
        })
        .collect_vec();

    let aux_evaluations = evaluation_points
        .iter()
        .map(|eval_point| {
            aux_trace_polys
                .iter()
                .map(|aux_poly| aux_poly.evaluate(eval_point))
                .collect_vec()
        })
        .collect_vec();

    debug_assert_eq!(main_evaluations.len(), aux_evaluations.len());
    let mut main_evaluations = main_evaluations;
    let mut table_data = Vec::new();
    for (main_row, aux_row) in main_evaluations.iter_mut().zip(aux_evaluations) {
        main_row.extend_from_slice(&aux_row);
        table_data.extend_from_slice(main_row);
    }

    let main_trace_width = main_trace_polys.len();
    let aux_trace_width = aux_trace_polys.len();
    let table_width = main_trace_width + aux_trace_width;

    Table::new(table_data, table_width)
}

pub fn columns2rows<F>(columns: Vec<Vec<F>>) -> Vec<Vec<F>>
where
    F: Clone,
{
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

/// Performs a fused bit-reverse permutation and column-to-row transpose.
/// This avoids the need to clone the columns for bit-reversal before transposing.
///
/// Instead of:
/// ```ignore
/// let mut permuted = columns.clone();  // Allocates full copy
/// for col in permuted.iter_mut() {
///     in_place_bit_reverse_permute(col);
/// }
/// let rows = columns2rows(permuted);
/// ```
///
/// Use:
/// ```ignore
/// let rows = columns2rows_bit_reversed(&columns);  // No clone needed
/// ```
pub fn columns2rows_bit_reversed<F>(columns: &[Vec<F>]) -> Vec<Vec<F>>
where
    F: Clone,
{
    if columns.is_empty() {
        return Vec::new();
    }

    use lambdaworks_math::fft::cpu::bit_reversing::reverse_index;

    let num_rows = columns[0].len();
    let num_cols = columns.len();

    (0..num_rows)
        .map(|row_index| {
            // Apply bit-reverse permutation while building the row
            let permuted_index = reverse_index(row_index, num_rows as u64);
            (0..num_cols)
                .map(|col_index| columns[col_index][permuted_index].clone())
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

    type FE = FieldElement<Stark252PrimeField>;

    #[test]
    fn test_compute_trace_polys_main_non_power_of_2_returns_error() {
        let non_pow2_column = vec![FE::one(); 7]; // 7 is not a power of 2
        let trace = TraceTable::<Stark252PrimeField, Stark252PrimeField>::from_columns_main(
            vec![non_pow2_column],
            1,
        );
        let result = trace.compute_trace_polys_main::<Stark252PrimeField>();
        assert!(
            result.is_err(),
            "Expected FFTError for non-power-of-2 trace length"
        );
    }
}
