#[derive(Clone)]
#[allow(dead_code)]
pub enum MdsMethod {
    /// Use standard matrix multiplication.
    MatrixMultiplication,
    /// Use Number Theoretic Transform for multiplication.
    Ntt,
    /// Use Karatsuba algorithm for multiplication.
    Karatsuba,
}
