use lambdaworks_math::errors::ByteConversionError;
use pyo3::{exceptions::PyValueError, PyErr};

/// Python wrapper for `ByteConversionError`. Provides an automattic conversion
/// between `ByteConversionError` into Python's `ValueError`.
pub struct PyByteConversionError(ByteConversionError);

impl From<ByteConversionError> for PyByteConversionError {
    fn from(error: ByteConversionError) -> Self {
        Self(error)
    }
}

impl From<PyByteConversionError> for PyErr {
    fn from(error: PyByteConversionError) -> Self {
        // `ValueError` is the idiomatic python exception type to rise when an
        // operation or function receives an argument that has the right type but an
        // inappropriate value.
        PyValueError::new_err(error.0.to_string())
    }
}
