//! Circuit DSL for building PLONK circuits with type safety.
//!
//! This module provides a high-level, type-safe API for constructing PLONK circuits.
//! It wraps the low-level constraint system with typed variables and composable gadgets.
//!
//! # Design Principles
//!
//! 1. **Type Safety**: Leverage Rust's type system for circuit correctness
//! 2. **Composability**: Gadgets compose cleanly without manual wiring
//! 3. **Readability**: Circuit code should read like the computation it proves
//! 4. **Performance**: Zero-cost abstractions where possible
//! 5. **Debugging**: Clear error messages pointing to constraint violations
//!
//! # Example
//!
//! ```ignore
//! use lambdaworks_plonk::dsl::CircuitBuilder;
//!
//! let mut builder = CircuitBuilder::new();
//!
//! // Create public inputs
//! let x = builder.public_input("x");
//! let y = builder.public_input("y");
//!
//! // Create private input
//! let secret = builder.private_input("secret");
//!
//! // Build circuit: x * secret == y
//! let product = builder.mul(&x, &secret);
//! builder.assert_eq(&product, &y);
//!
//! // Build the circuit
//! let circuit = builder.build();
//! ```

pub mod builder;
pub mod gadgets;
pub mod types;

pub use builder::CircuitBuilder;
pub use types::{BoolVar, FieldVar, Var};
