//! Typed variable representations for the circuit DSL.
//!
//! This module provides type-safe wrappers around circuit variables,
//! enabling compile-time checking of circuit operations.

use crate::constraint_system::Variable;
use std::marker::PhantomData;

/// A typed circuit variable.
///
/// `Var<T>` wraps a raw `Variable` with a type marker to enable
/// type-safe operations. The type parameter `T` indicates what
/// kind of value the variable represents (field element, boolean, etc.).
#[derive(Clone, Copy, Debug)]
pub struct Var<T> {
    /// The underlying variable ID in the constraint system
    pub(crate) inner: Variable,
    /// Phantom data for the type marker
    _type: PhantomData<T>,
}

impl<T> Var<T> {
    /// Creates a new typed variable from a raw variable.
    pub(crate) fn new(inner: Variable) -> Self {
        Self {
            inner,
            _type: PhantomData,
        }
    }

    /// Returns the underlying variable ID.
    pub fn variable(&self) -> Variable {
        self.inner
    }
}

impl<T> PartialEq for Var<T> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<T> Eq for Var<T> {}

impl<T> std::hash::Hash for Var<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

/// Marker type for field element variables.
///
/// Variables of type `Var<FieldType>` represent arbitrary field elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldType;

/// Marker type for boolean variables.
///
/// Variables of type `Var<BoolType>` are constrained to be 0 or 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoolType;

/// Marker type for unsigned 8-bit integers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct U8Type;

/// Marker type for unsigned 32-bit integers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct U32Type;

/// Marker type for unsigned 64-bit integers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct U64Type;

/// A field element variable.
pub type FieldVar = Var<FieldType>;

/// A boolean variable (constrained to 0 or 1).
pub type BoolVar = Var<BoolType>;

/// An 8-bit unsigned integer variable.
pub type U8Var = Var<U8Type>;

/// A 32-bit unsigned integer variable.
pub type U32Var = Var<U32Type>;

/// A 64-bit unsigned integer variable.
pub type U64Var = Var<U64Type>;

/// Trait for types that can be converted to a field variable.
///
/// This enables operations that accept multiple variable types.
pub trait AsFieldVar {
    /// Converts to a field variable.
    fn as_field_var(&self) -> FieldVar;
}

impl AsFieldVar for FieldVar {
    fn as_field_var(&self) -> FieldVar {
        *self
    }
}

impl AsFieldVar for BoolVar {
    fn as_field_var(&self) -> FieldVar {
        FieldVar::new(self.inner)
    }
}

impl AsFieldVar for U8Var {
    fn as_field_var(&self) -> FieldVar {
        FieldVar::new(self.inner)
    }
}

impl AsFieldVar for U32Var {
    fn as_field_var(&self) -> FieldVar {
        FieldVar::new(self.inner)
    }
}

impl AsFieldVar for U64Var {
    fn as_field_var(&self) -> FieldVar {
        FieldVar::new(self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_var_equality() {
        let v1: FieldVar = Var::new(1);
        let v2: FieldVar = Var::new(1);
        let v3: FieldVar = Var::new(2);

        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_var_variable_id() {
        let v: FieldVar = Var::new(42);
        assert_eq!(v.variable(), 42);
    }

    #[test]
    fn test_as_field_var() {
        let field_var: FieldVar = Var::new(1);
        let bool_var: BoolVar = Var::new(2);

        assert_eq!(field_var.as_field_var().variable(), 1);
        assert_eq!(bool_var.as_field_var().variable(), 2);
    }
}
