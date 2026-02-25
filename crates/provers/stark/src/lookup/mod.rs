//! # LogUp Lookup Argument with Circular Transition Constraints
//!
//! This module implements the **LogUp** (Logarithmic Derivative) lookup argument
//! for the STARK prover, using a **circular transition constraint** that eliminates
//! all LogUp-specific boundary constraints.
//!
//! ## Background: LogUp
//!
//! LogUp ([Haböck 2023](https://eprint.iacr.org/2023/1518)) reduces lookup arguments
//! to a sum-check over logarithmic derivatives. For each bus interaction (sender or
//! receiver), we compute a **term column**:
//!
//! ```text
//! term[i] = sign * multiplicity[i] / fingerprint[i]
//! ```
//!
//! where `fingerprint[i] = z - (bus_id + α·v₀ + α²·v₁ + ...)` is a random linear
//! combination of the row values, and `sign = +1` for senders, `-1` for receivers.
//!
//! An **accumulated column** tracks the running sum of all terms. If senders and
//! receivers are balanced, the total sum is zero.
//!
//! ## Circular Transition Constraint
//!
//! The standard approach pins the accumulated column with boundary constraints:
//! fix `acc[0]` and `acc[N-1]`, then enforce `acc[i+1] - acc[i] = Σ terms[i+1]`
//! for rows `0..N-2` (with one row exempted at the end). This requires multiple
//! boundary constraints and public inputs carrying initial term values and the
//! final accumulated value.
//!
//! We instead use a **circular transition constraint**, following the approach
//! used in [Stwo](https://github.com/starkware-libs/stwo) by StarkWare. The key
//! insight is that in STARKs, transition constraints are evaluated over polynomials,
//! and with `transition_offsets = [0, 1]` and `end_exemptions = 0`, the evaluation
//! at row `N-1` naturally wraps to row `0` via the polynomial's periodicity. This
//! lets us write a single constraint that holds for **all** rows including the wrap:
//!
//! ```text
//! acc[(i+1) mod N] - acc[i] - Σ terms[(i+1) mod N] + L/N = 0
//! ```
//!
//! where:
//! - `L = Σ_i Σ_k term_k[i]` is the total sum of all terms across all rows and columns
//! - `N` is the trace length
//! - `L/N` is a constant offset subtracted each row so the column wraps circularly
//!
//! ### Why it works
//!
//! We build the accumulated column as:
//!
//! ```text
//! acc[0] = row_sum[0] - L/N
//! acc[i] = acc[i-1] + row_sum[i] - L/N    for i > 0
//! ```
//!
//! Telescoping: `acc[N-1] = Σ row_sums - N·(L/N) = L - L = 0`. Since `acc[0]`
//! wraps from `acc[N-1] = 0`, the transition constraint at row `N-1` becomes:
//!
//! ```text
//! acc[0] - 0 - row_sum[0] + L/N = 0  ✓  (by construction of acc[0])
//! ```
//!
//! This eliminates **all** LogUp-specific boundary constraints. The only public
//! input needed is `L` (the `table_contribution`), which for cross-table bus
//! balance must satisfy `Σ L_table = 0` across all participating tables.
//!
//! ### Soundness
//!
//! If a malicious prover tampers with `L`, the circular constraint evaluation at
//! the out-of-domain (OOD) point will fail verification — the verifier recomputes
//! `L/N` from the claimed `table_contribution` and checks it against the committed
//! trace polynomials. No additional boundary constraints are needed to ensure
//! soundness.
//!
//! ## Module Structure
//!
//! - [`types`]: Core types (`BusInteraction`, `BusValue`, `Multiplicity`, `BusPublicInputs`)
//! - [`trace_builder`]: Builds term columns (with batch inversion) and the circular
//!   accumulated column
//! - [`constraints`]: Transition constraints for term verification and accumulation
//! - [`air`]: `AirWithLogUp` — wraps user-defined AIR with automatic LogUp machinery

pub mod air;
pub mod constraints;
pub mod trace_builder;
pub mod types;

pub use air::AirWithLogUp;
pub use types::*;
