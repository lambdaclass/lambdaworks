use lambdaworks_math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};

use crate::constraints::boundary::BoundaryConstraint;

// =============================================================================
// LogUp Challenge Indices
// =============================================================================

/// Index of the `z` challenge in the LogUp challenges vector.
/// Used as the evaluation point in fingerprint computation.
pub const LOGUP_CHALLENGE_Z: usize = 0;

/// Index of the `alpha` (α) challenge in the LogUp challenges vector.
/// Used as the base for linear combination of row values.
pub const LOGUP_CHALLENGE_ALPHA: usize = 1;

/// Number of challenges required by the LogUp protocol.
pub const LOGUP_NUM_CHALLENGES: usize = 2;

// =============================================================================
// Linear Term and Bus Value
// =============================================================================

/// A term in a linear combination of column values and constants.
#[derive(Debug, Clone)]
pub enum LinearTerm {
    /// coefficient * column_value (coefficient can be negative)
    Column {
        /// The multiplier for the column value (signed to support subtraction)
        coefficient: i64,
        /// The column index to read from
        column: usize,
    },
    /// coefficient * column_value (unsigned, for large field elements like inverses)
    ColumnUnsigned {
        /// The multiplier as an unsigned value (for large field elements)
        coefficient: u64,
        /// The column index to read from
        column: usize,
    },
    /// A constant value to add (signed to support subtraction)
    Constant(i64),
    /// A constant value to add (unsigned, for large values that don't fit in i64)
    ConstantUnsigned(u64),
}

/// A value that contributes to the bus fingerprint.
///
/// Each `BusValue` produces exactly **1 bus element** for the fingerprint.
/// The fingerprint is computed as: `z - (v₀ + α·v₁ + α²·v₂ + ...)`
/// where each `vᵢ` is a bus element from a `BusValue`.
#[derive(Debug, Clone)]
pub enum BusValue {
    /// A single column value.
    Column(usize),
    /// Custom linear combination of columns and/or constants.
    Linear(Vec<LinearTerm>),
}

impl BusValue {
    /// Creates a constant value (no columns).
    pub fn constant(value: u64) -> Self {
        BusValue::Linear(vec![LinearTerm::ConstantUnsigned(value)])
    }

    /// Creates a single column value.
    pub fn column(col: usize) -> Self {
        BusValue::Column(col)
    }

    /// Creates a linear combination from terms.
    pub fn linear(terms: Vec<LinearTerm>) -> Self {
        BusValue::Linear(terms)
    }

    /// Creates BusValues for multiple columns (each as a separate Column variant).
    pub fn columns(cols: &[usize]) -> Vec<BusValue> {
        cols.iter().map(|&col| BusValue::Column(col)).collect()
    }

    /// Computes the bus element value from column values.
    pub fn combine_from<E: IsField, G: Fn(usize) -> FieldElement<E>>(
        &self,
        get_column: G,
    ) -> FieldElement<E> {
        match self {
            BusValue::Column(col) => get_column(*col),
            BusValue::Linear(terms) => {
                let mut result = FieldElement::<E>::zero();
                for term in terms {
                    match term {
                        LinearTerm::Column {
                            coefficient,
                            column,
                        } => {
                            let coeff = if *coefficient >= 0 {
                                FieldElement::<E>::from(*coefficient as u64)
                            } else {
                                -FieldElement::<E>::from(coefficient.unsigned_abs())
                            };
                            result += get_column(*column) * coeff;
                        }
                        LinearTerm::ColumnUnsigned {
                            coefficient,
                            column,
                        } => {
                            let coeff = FieldElement::<E>::from(*coefficient);
                            result += get_column(*column) * coeff;
                        }
                        LinearTerm::Constant(value) => {
                            if *value >= 0 {
                                result += FieldElement::<E>::from(*value as u64);
                            } else {
                                result = result - FieldElement::<E>::from(value.unsigned_abs());
                            }
                        }
                        LinearTerm::ConstantUnsigned(value) => {
                            result += FieldElement::<E>::from(*value);
                        }
                    }
                }
                result
            }
        }
    }
}

// =============================================================================
// Multiplicity
// =============================================================================

/// Specifies how to compute the multiplicity for a bus interaction.
///
/// **Important**: The LogUp module reads multiplicity values from the main trace
/// but does NOT constrain them. Your main AIR must include transition or boundary
/// constraints that ensure multiplicity columns contain correct values (e.g.,
/// range checks, boolean checks, or consistency with actual occurrence counts).
/// Without these constraints, a malicious prover could set arbitrary multiplicities
/// and break bus balance soundness.
#[derive(Clone, Debug)]
pub enum Multiplicity {
    /// Constant multiplicity of 1 for all rows.
    One,
    /// Read multiplicity from a single column.
    Column(usize),
    /// Sum of two columns: `col_a + col_b`.
    Sum(usize, usize),
    /// Negation of a bit column: `1 - col_value`.
    Negated(usize),
    /// Arbitrary linear combination of columns and constants.
    Linear(Vec<LinearTerm>),
}

// =============================================================================
// Bus Interaction
// =============================================================================

/// A single bus interaction (sender or receiver).
///
/// The `bus_id` distinguishes different buses. Senders and receivers must use
/// the same `bus_id` for their fingerprints to match.
#[derive(Clone)]
pub struct BusInteraction {
    /// Bus identifier.
    pub bus_id: u64,
    /// How to compute the multiplicity for this interaction.
    pub multiplicity: Multiplicity,
    /// Bus values that make up this interaction.
    pub values: Vec<BusValue>,
    /// Whether this side of the interaction is a sender (true) or receiver (false).
    pub is_sender: bool,
}

impl BusInteraction {
    /// Creates a new bus interaction.
    pub fn new(
        bus_id: impl Into<u64>,
        multiplicity: Multiplicity,
        values: Vec<BusValue>,
        is_sender: bool,
    ) -> Self {
        Self {
            bus_id: bus_id.into(),
            multiplicity,
            values,
            is_sender,
        }
    }

    /// Creates a sender interaction.
    pub fn sender(
        bus_id: impl Into<u64>,
        multiplicity: Multiplicity,
        values: Vec<BusValue>,
    ) -> Self {
        Self::new(bus_id, multiplicity, values, true)
    }

    /// Creates a receiver interaction.
    pub fn receiver(
        bus_id: impl Into<u64>,
        multiplicity: Multiplicity,
        values: Vec<BusValue>,
    ) -> Self {
        Self::new(bus_id, multiplicity, values, false)
    }

    /// Returns total number of bus elements (for α power computation).
    /// Includes the bus_id as the first element.
    pub fn num_bus_elements(&self) -> usize {
        1 + self.values.len()
    }
}

// =============================================================================
// Bus Public Inputs
// =============================================================================

/// Public inputs for a table's accumulated LogUp column.
///
/// Note: `initial_value` is always zero (hardcoded verifier-known constant).
/// It is kept here for documentation purposes; in practice only
/// `final_accumulated` is needed for the cross-table bus balance check.
#[derive(Debug, Clone)]
pub struct BusPublicInputs<E: IsField> {
    /// Accumulated column value at row 0 (always zero — verifier-known constant).
    pub initial_value: FieldElement<E>,
    /// Accumulated column value at last row.
    pub final_accumulated: FieldElement<E>,
}

// =============================================================================
// Auxiliary Trace Build Data
// =============================================================================

/// Container for interaction data needed to build aux trace.
pub struct AuxiliaryTraceBuildData {
    pub interactions: Vec<BusInteraction>,
}

// =============================================================================
// Boundary Constraint Builder
// =============================================================================

/// Trait for user-defined boundary constraints alongside LogUp ones.
pub trait BoundaryConstraintBuilder<
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
    PI,
>: Send + Sync
{
    fn boundary_constraints(
        _pub_inputs: &PI,
        _rap_challenges: &[FieldElement<E>],
    ) -> Vec<BoundaryConstraint<E>> {
        vec![]
    }
}

/// No-op implementor when no extra boundary constraints are needed.
pub struct NullBoundaryConstraintBuilder;

impl<F, E, PI> BoundaryConstraintBuilder<F, E, PI> for NullBoundaryConstraintBuilder
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
}
