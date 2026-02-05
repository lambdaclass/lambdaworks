/// Implementation of FieldElement, a generic element of a field.
pub mod element;
pub mod errors;
/// Implementation of quadratic extensions of fields.
pub mod extensions;
/// Implementation of particular cases of fields.
pub mod fields;
/// Field for test purposes.
pub mod test_fields;
/// Test helpers and macros for field testing.
#[cfg(test)]
pub mod test_helpers;
/// Common behaviour for field elements.
pub mod traits;
