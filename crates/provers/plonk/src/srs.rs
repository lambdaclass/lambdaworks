//! SRS (Structured Reference String) management for PLONK.
//!
//! This module provides utilities for loading, saving, and validating SRS files.
//! The SRS is generated through a trusted setup ceremony and contains the
//! powers of a secret element needed for KZG polynomial commitments.
//!
//! # Security
//!
//! The security of PLONK relies on the assumption that the "toxic waste"
//! (the secret element used to generate the SRS) has been securely destroyed.
//! In production, always use SRS from trusted ceremonies (e.g., Zcash Powers of Tau).
//!
//! # Format
//!
//! This module uses a simple binary format with version tagging for forward
//! compatibility. For interoperability with other systems, consider using
//! standard formats like the Zcash Powers of Tau format.

use lambdaworks_crypto::commitments::kzg::StructuredReferenceString;
use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::errors::DeserializationError;
use lambdaworks_math::traits::{AsBytes, Deserializable};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Number of G2 powers in standard KZG SRS (g2, s*g2)
const G2_POWERS_COUNT: usize = 2;

/// Errors that can occur during SRS operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SRSError {
    /// File I/O error
    IoError(String),
    /// Deserialization failed
    DeserializationError(String),
    /// SRS validation failed
    ValidationFailed(String),
    /// SRS is too small for the circuit
    InsufficientSize { required: usize, available: usize },
    /// Invalid file format or version
    InvalidFormat(String),
}

impl std::fmt::Display for SRSError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SRSError::IoError(msg) => write!(f, "I/O error: {}", msg),
            SRSError::DeserializationError(msg) => write!(f, "Deserialization error: {}", msg),
            SRSError::ValidationFailed(msg) => write!(f, "SRS validation failed: {}", msg),
            SRSError::InsufficientSize {
                required,
                available,
            } => {
                write!(
                    f,
                    "SRS too small: circuit requires {} powers, SRS has {}",
                    required, available
                )
            }
            SRSError::InvalidFormat(msg) => write!(f, "Invalid SRS format: {}", msg),
        }
    }
}

impl std::error::Error for SRSError {}

impl From<std::io::Error> for SRSError {
    fn from(err: std::io::Error) -> Self {
        SRSError::IoError(err.to_string())
    }
}

impl From<DeserializationError> for SRSError {
    fn from(err: DeserializationError) -> Self {
        SRSError::DeserializationError(format!("{:?}", err))
    }
}

/// SRS file format version.
const SRS_FORMAT_VERSION: u8 = 1;

/// Manages SRS loading, saving, and validation.
///
/// # Example
///
/// ```ignore
/// use lambdaworks_plonk::srs::SRSManager;
///
/// // Load SRS from file
/// let srs = SRSManager::load("ceremony.srs")?;
///
/// // Validate SRS consistency
/// SRSManager::validate(&srs)?;
///
/// // Check if SRS is large enough for circuit
/// SRSManager::check_size(&srs, circuit_size)?;
/// ```
pub struct SRSManager;

impl SRSManager {
    /// Loads an SRS from a file.
    ///
    /// # Arguments
    /// * `path` - Path to the SRS file
    ///
    /// # Returns
    /// * `Ok(StructuredReferenceString)` on success
    /// * `Err(SRSError)` if loading fails
    pub fn load<G1, G2, P: AsRef<Path>>(
        path: P,
    ) -> Result<StructuredReferenceString<G1, G2>, SRSError>
    where
        G1: Deserializable + IsGroup,
        G2: Deserializable + IsGroup,
    {
        let mut file = File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        Self::from_bytes(&bytes)
    }

    /// Saves an SRS to a file.
    ///
    /// # Arguments
    /// * `srs` - The SRS to save
    /// * `path` - Path where the SRS will be saved
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(SRSError)` if saving fails
    pub fn save<G1, G2, P: AsRef<Path>>(
        srs: &StructuredReferenceString<G1, G2>,
        path: P,
    ) -> Result<(), SRSError>
    where
        G1: AsBytes,
        G2: AsBytes,
    {
        let bytes = Self::to_bytes(srs);
        let mut file = File::create(path)?;
        file.write_all(&bytes)?;
        Ok(())
    }

    /// Serializes an SRS to bytes.
    pub fn to_bytes<G1, G2>(srs: &StructuredReferenceString<G1, G2>) -> Vec<u8>
    where
        G1: AsBytes,
        G2: AsBytes,
    {
        let mut bytes = Vec::new();

        // Version byte
        bytes.push(SRS_FORMAT_VERSION);

        // Number of G1 powers (u64)
        let g1_powers = srs.powers_main_group.len();
        bytes.extend_from_slice(&(g1_powers as u64).to_be_bytes());

        // Serialize G1 powers with length prefix
        for point in &srs.powers_main_group {
            let point_bytes = point.as_bytes();
            bytes.extend_from_slice(&(point_bytes.len() as u32).to_be_bytes());
            bytes.extend_from_slice(&point_bytes);
        }

        // Number of G2 powers (should be 2 for KZG: [g2, s*g2])
        let g2_powers = srs.powers_secondary_group.len();
        bytes.extend_from_slice(&(g2_powers as u64).to_be_bytes());

        // Serialize G2 powers with length prefix
        for point in &srs.powers_secondary_group {
            let point_bytes = point.as_bytes();
            bytes.extend_from_slice(&(point_bytes.len() as u32).to_be_bytes());
            bytes.extend_from_slice(&point_bytes);
        }

        bytes
    }

    /// Deserializes an SRS from bytes.
    pub fn from_bytes<G1, G2>(bytes: &[u8]) -> Result<StructuredReferenceString<G1, G2>, SRSError>
    where
        G1: Deserializable + IsGroup,
        G2: Deserializable + IsGroup,
    {
        if bytes.is_empty() {
            return Err(SRSError::InvalidFormat("Empty SRS data".to_string()));
        }

        // Check version
        let version = bytes[0];
        if version != SRS_FORMAT_VERSION {
            return Err(SRSError::InvalidFormat(format!(
                "Unsupported SRS version: {} (expected {})",
                version, SRS_FORMAT_VERSION
            )));
        }

        let mut offset = 1;

        // Read G1 powers count
        if offset + 8 > bytes.len() {
            return Err(SRSError::InvalidFormat("Truncated G1 count".to_string()));
        }
        let g1_count = u64::from_be_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        // Read G1 powers
        let mut powers_main_group = Vec::with_capacity(g1_count);
        for _ in 0..g1_count {
            if offset + 4 > bytes.len() {
                return Err(SRSError::InvalidFormat(
                    "Truncated G1 point length".to_string(),
                ));
            }
            let point_len =
                u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;

            if offset + point_len > bytes.len() {
                return Err(SRSError::InvalidFormat(
                    "Truncated G1 point data".to_string(),
                ));
            }
            let point = G1::deserialize(&bytes[offset..offset + point_len])?;
            powers_main_group.push(point);
            offset += point_len;
        }

        // Read G2 powers count
        if offset + 8 > bytes.len() {
            return Err(SRSError::InvalidFormat("Truncated G2 count".to_string()));
        }
        let g2_count = u64::from_be_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        // KZG requires exactly 2 G2 powers
        if g2_count != G2_POWERS_COUNT {
            return Err(SRSError::InvalidFormat(format!(
                "Expected {} G2 powers, got {}",
                G2_POWERS_COUNT, g2_count
            )));
        }

        // Read G2 powers into a fixed-size array
        let mut g2_points: Vec<G2> = Vec::with_capacity(G2_POWERS_COUNT);
        for _ in 0..G2_POWERS_COUNT {
            if offset + 4 > bytes.len() {
                return Err(SRSError::InvalidFormat(
                    "Truncated G2 point length".to_string(),
                ));
            }
            let point_len =
                u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;

            if offset + point_len > bytes.len() {
                return Err(SRSError::InvalidFormat(
                    "Truncated G2 point data".to_string(),
                ));
            }
            let point = G2::deserialize(&bytes[offset..offset + point_len])?;
            g2_points.push(point);
            offset += point_len;
        }

        // Convert Vec to fixed-size array
        let powers_secondary_group: [G2; 2] = g2_points.try_into().map_err(|_| {
            SRSError::InvalidFormat("Failed to convert G2 powers to array".to_string())
        })?;

        Ok(StructuredReferenceString::new(
            &powers_main_group,
            &powers_secondary_group,
        ))
    }

    /// Checks if an SRS is large enough for a given circuit size.
    ///
    /// For PLONK, the SRS needs at least `n + 3` G1 powers where `n` is the
    /// number of constraints (after padding to power of 2).
    ///
    /// # Arguments
    /// * `srs` - The SRS to check
    /// * `circuit_size` - Number of constraints (will be padded internally)
    ///
    /// # Returns
    /// * `Ok(())` if the SRS is large enough
    /// * `Err(SRSError::InsufficientSize)` if the SRS is too small
    pub fn check_size<G1, G2>(
        srs: &StructuredReferenceString<G1, G2>,
        circuit_size: usize,
    ) -> Result<(), SRSError> {
        // PLONK requires n + 3 powers for commitment
        let padded_size = circuit_size.next_power_of_two();
        let required = padded_size + 3;
        let available = srs.powers_main_group.len();

        if available < required {
            return Err(SRSError::InsufficientSize {
                required,
                available,
            });
        }

        Ok(())
    }

    // Note: SRS validation using pairing checks would go here.
    // This requires proper pairing trait bounds which depend on the specific curve.
    // For BLS12-381, use the BLS12381AtePairing type.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::utils::{test_srs, G1Point, G2Point};

    #[test]
    fn test_srs_serialization_roundtrip() {
        let srs = test_srs(8);

        // Serialize
        let bytes = SRSManager::to_bytes(&srs);

        // Check version byte
        assert_eq!(bytes[0], SRS_FORMAT_VERSION);

        // Deserialize
        let deserialized: StructuredReferenceString<G1Point, G2Point> =
            SRSManager::from_bytes(&bytes).expect("Deserialization failed");

        // Verify
        assert_eq!(
            srs.powers_main_group.len(),
            deserialized.powers_main_group.len()
        );
        assert_eq!(
            srs.powers_secondary_group.len(),
            deserialized.powers_secondary_group.len()
        );

        for (a, b) in srs
            .powers_main_group
            .iter()
            .zip(deserialized.powers_main_group.iter())
        {
            assert_eq!(a, b);
        }
        for (a, b) in srs
            .powers_secondary_group
            .iter()
            .zip(deserialized.powers_secondary_group.iter())
        {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_srs_check_size_sufficient() {
        let srs = test_srs(16); // Creates SRS with n+3 = 19 powers
        let result = SRSManager::check_size(&srs, 16);
        assert!(result.is_ok());
    }

    #[test]
    fn test_srs_check_size_insufficient() {
        let srs = test_srs(4); // Creates SRS with 7 powers
        let result = SRSManager::check_size(&srs, 16); // Needs 19 powers
        assert!(result.is_err());
        assert!(matches!(result, Err(SRSError::InsufficientSize { .. })));
    }

    #[test]
    fn test_srs_from_bytes_empty() {
        let result: Result<StructuredReferenceString<G1Point, G2Point>, _> =
            SRSManager::from_bytes(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_srs_from_bytes_wrong_version() {
        let srs = test_srs(4);
        let mut bytes = SRSManager::to_bytes(&srs);
        bytes[0] = 99; // Invalid version

        let result: Result<StructuredReferenceString<G1Point, G2Point>, _> =
            SRSManager::from_bytes(&bytes);
        assert!(result.is_err());
        assert!(matches!(result, Err(SRSError::InvalidFormat(_))));
    }

    #[test]
    fn test_srs_file_roundtrip() {
        use std::env::temp_dir;

        let srs = test_srs(8);
        let path = temp_dir().join("test_srs.bin");

        // Save
        SRSManager::save(&srs, &path).expect("Failed to save SRS");

        // Load
        let loaded: StructuredReferenceString<G1Point, G2Point> =
            SRSManager::load(&path).expect("Failed to load SRS");

        // Verify
        assert_eq!(srs.powers_main_group.len(), loaded.powers_main_group.len());

        // Cleanup
        std::fs::remove_file(&path).ok();
    }
}
