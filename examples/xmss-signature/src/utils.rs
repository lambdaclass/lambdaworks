//! Utility functions for XMSS
//!
//! This module contains helper functions used throughout the XMSS implementation,
//! most notably the base_w conversion function for Winternitz encoding.

use crate::params::{LEN, LEN_1, LEN_2, W};

/// Convert a byte array to base-w representation
///
/// This function takes a message (byte array) and converts it to an array of
/// base-w digits. For w=16, each byte is split into 2 nibbles (4 bits each).
///
/// # Arguments
/// * `input` - The input byte array to convert
/// * `out_len` - The number of base-w digits to produce
///
/// # Returns
/// A vector of base-w digits
pub fn base_w(input: &[u8], out_len: usize) -> Vec<u32> {
    let log_w = match W {
        4 => 2,
        16 => 4,
        256 => 8,
        _ => (W as f64).log2() as u32,
    };

    let mut result = vec![0u32; out_len];
    let mut in_idx = 0;
    let mut bits = 0u32;
    let mut total = 0u32;

    for item in result.iter_mut() {
        if bits == 0 {
            if in_idx < input.len() {
                total = input[in_idx] as u32;
                in_idx += 1;
            } else {
                total = 0;
            }
            bits = 8;
        }
        bits -= log_w;
        *item = (total >> bits) & ((W as u32) - 1);
    }

    result
}

/// Compute the checksum for WOTS+ message
///
/// The checksum ensures that an attacker cannot forge signatures by
/// only computing forward in the hash chains.
///
/// # Arguments
/// * `msg_base_w` - The message in base-w representation (LEN_1 digits)
///
/// # Returns
/// The checksum in base-w representation (LEN_2 digits)
pub fn compute_checksum(msg_base_w: &[u32]) -> Vec<u32> {
    let mut csum: u32 = 0;

    // Sum up (w - 1 - msg[i]) for all message digits
    for &digit in msg_base_w.iter().take(LEN_1) {
        csum += (W as u32) - 1 - digit;
    }

    // Left-shift checksum by 4 bits (for w=16, to align to byte boundary)
    // csum << (8 - ((len_2 * log_w) % 8)) % 8
    let log_w = 4u32; // for w=16
    let shift = (8 - ((LEN_2 as u32 * log_w) % 8)) % 8;
    csum <<= shift;

    // Convert checksum to bytes then to base-w
    let csum_bytes = csum.to_be_bytes();
    // We need LEN_2 digits, which requires ceil(LEN_2 * log_w / 8) bytes
    let needed_bytes = (LEN_2 * 4).div_ceil(8);
    let start = 4 - needed_bytes;
    base_w(&csum_bytes[start..], LEN_2)
}

/// Convert message to full WOTS+ input (message + checksum)
///
/// # Arguments
/// * `msg_hash` - The hashed message (32 bytes)
///
/// # Returns
/// A vector of LEN base-w digits (message digits + checksum digits)
pub fn msg_to_wots_input(msg_hash: &[u8; 32]) -> Vec<u32> {
    let mut result = Vec::with_capacity(LEN);

    // Convert message hash to base-w
    let msg_base_w = base_w(msg_hash, LEN_1);
    result.extend_from_slice(&msg_base_w);

    // Compute and append checksum
    let checksum = compute_checksum(&msg_base_w);
    result.extend_from_slice(&checksum);

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_w_zeros() {
        let input = [0u8; 32];
        let result = base_w(&input, 64);
        assert_eq!(result.len(), 64);
        assert!(result.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_base_w_ones() {
        // 0xFF in base-16 is [15, 15]
        let input = [0xFF];
        let result = base_w(&input, 2);
        assert_eq!(result, vec![15, 15]);
    }

    #[test]
    fn test_base_w_mixed() {
        // 0xAB = 10*16 + 11 = [10, 11] in base-16
        let input = [0xAB];
        let result = base_w(&input, 2);
        assert_eq!(result, vec![10, 11]);
    }

    #[test]
    fn test_base_w_multiple_bytes() {
        // 0x12 0x34 = [1, 2, 3, 4] in base-16
        let input = [0x12, 0x34];
        let result = base_w(&input, 4);
        assert_eq!(result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_checksum_zeros() {
        // All zeros means maximum checksum (64 * 15 = 960)
        let msg_base_w = vec![0u32; LEN_1];
        let checksum = compute_checksum(&msg_base_w);
        assert_eq!(checksum.len(), LEN_2);
    }

    #[test]
    fn test_checksum_max() {
        // All 15s means minimum checksum (0)
        let msg_base_w = vec![15u32; LEN_1];
        let checksum = compute_checksum(&msg_base_w);
        assert_eq!(checksum.len(), LEN_2);
        // Checksum should be [0, 0, 0] for max message values
        assert!(checksum.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_msg_to_wots_input_length() {
        let msg_hash = [0u8; 32];
        let result = msg_to_wots_input(&msg_hash);
        assert_eq!(result.len(), LEN);
    }
}
