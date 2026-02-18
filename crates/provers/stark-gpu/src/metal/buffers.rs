/// A typed GPU buffer wrapping a Metal buffer that holds polynomial coefficients
/// or evaluations.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct PolyBuffer {
    /// The underlying Metal buffer.
    pub buffer: metal::Buffer,
    /// Number of field elements stored in the buffer.
    pub len: usize,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl PolyBuffer {
    /// Creates a new `PolyBuffer` wrapping an existing Metal buffer.
    pub fn new(buffer: metal::Buffer, len: usize) -> Self {
        Self { buffer, len }
    }

    /// Returns the number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A typed GPU buffer wrapping a Metal buffer that holds a Low-Degree Extension
/// (LDE) matrix, stored as a flat array in row-major order.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct LdeBuffer {
    /// The underlying Metal buffer.
    pub buffer: metal::Buffer,
    /// Number of columns (traces) in the LDE matrix.
    pub num_cols: usize,
    /// Number of rows (evaluation points) in the LDE matrix.
    pub num_rows: usize,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl LdeBuffer {
    /// Creates a new `LdeBuffer` wrapping an existing Metal buffer.
    pub fn new(buffer: metal::Buffer, num_cols: usize, num_rows: usize) -> Self {
        Self {
            buffer,
            num_cols,
            num_rows,
        }
    }

    /// Returns the number of columns in the LDE matrix.
    pub fn num_cols(&self) -> usize {
        self.num_cols
    }

    /// Returns the number of rows in the LDE matrix.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Returns the total number of elements in the LDE matrix.
    pub fn total_elements(&self) -> usize {
        self.num_cols * self.num_rows
    }
}

// Stubs for non-Metal platforms

/// Stub `PolyBuffer` for non-Metal platforms.
#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub struct PolyBuffer {
    _private: (),
}

/// Stub `LdeBuffer` for non-Metal platforms.
#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub struct LdeBuffer {
    _private: (),
}
