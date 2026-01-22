//! XMSS Parameters as defined in RFC 8391
//!
//! This module defines the parameter sets for XMSS signatures.
//! Default parameters use SHA-256 with n=32, w=16, h=10.

/// Hash output size in bytes (SHA-256)
pub const N: usize = 32;

/// Winternitz parameter (base for encoding)
/// w=16 means 4 bits per digit, balancing signature size and computation
pub const W: usize = 16;

/// Tree height - determines number of available signatures (2^h)
/// h=10 allows 1024 one-time signatures
pub const H: usize = 10;

/// Number of chains for message hash
/// len_1 = ceil(8n / log2(w)) = ceil(256 / 4) = 64
pub const LEN_1: usize = 64;

/// Number of chains for checksum
/// len_2 = floor(log2(len_1 * (w-1)) / log2(w)) + 1 = floor(log2(64*15)/4) + 1 = 3
pub const LEN_2: usize = 3;

/// Total number of WOTS+ chains
/// len = len_1 + len_2 = 67
pub const LEN: usize = LEN_1 + LEN_2;

/// Maximum index value (2^h - 1)
pub const MAX_IDX: u32 = (1 << H) - 1;

/// XMSS parameter set
#[derive(Debug, Clone, Copy)]
pub struct XmssParams {
    /// Security parameter / hash output length in bytes
    pub n: usize,
    /// Winternitz parameter
    pub w: usize,
    /// Tree height
    pub h: usize,
    /// Message chains count
    pub len_1: usize,
    /// Checksum chains count
    pub len_2: usize,
    /// Total chains count
    pub len: usize,
}

impl Default for XmssParams {
    fn default() -> Self {
        Self {
            n: N,
            w: W,
            h: H,
            len_1: LEN_1,
            len_2: LEN_2,
            len: LEN,
        }
    }
}

impl XmssParams {
    /// Create a new parameter set with custom values
    pub fn new(n: usize, w: usize, h: usize) -> Self {
        // Calculate len_1 and len_2 based on n and w
        let log_w = (w as f64).log2() as usize;
        let len_1 = (8 * n + log_w - 1) / log_w; // ceil(8n / log2(w))
        let len_2 = ((((len_1 * (w - 1)) as f64).log2() / log_w as f64).floor() as usize) + 1;
        let len = len_1 + len_2;

        Self {
            n,
            w,
            h,
            len_1,
            len_2,
            len,
        }
    }

    /// Returns the number of available signatures (2^h)
    pub fn num_signatures(&self) -> u64 {
        1u64 << self.h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let params = XmssParams::default();
        assert_eq!(params.n, 32);
        assert_eq!(params.w, 16);
        assert_eq!(params.h, 10);
        assert_eq!(params.len_1, 64);
        assert_eq!(params.len_2, 3);
        assert_eq!(params.len, 67);
    }

    #[test]
    fn test_num_signatures() {
        let params = XmssParams::default();
        assert_eq!(params.num_signatures(), 1024);
    }

    #[test]
    fn test_custom_params() {
        let params = XmssParams::new(32, 16, 5);
        assert_eq!(params.h, 5);
        assert_eq!(params.num_signatures(), 32);
    }
}
