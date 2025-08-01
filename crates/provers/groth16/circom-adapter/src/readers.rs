use lambdaworks_groth16::common::FrElement;
use lambdaworks_math::{errors::CreationError, unsigned_integer::element::UnsignedInteger};
use serde::Deserialize;
use std::{collections::HashMap, fs::File, io::BufReader, path::Path};

#[derive(Debug)]
pub enum CircomReaderError {
    IOError(std::io::Error),
    SerdeError(serde_json::Error),
    ParseError(CreationError),
}

/// A witness for a Circom circuit, alias for a vector of field elements.
pub type CircomWitness = Vec<FrElement>;

/// A rank-1 constraint system (R1CS) for a Circom circuit, as read from
/// a JSON export of that R1CS.
///
/// Further information:
/// - <https://github.com/iden3/r1csfile/blob/master/doc/r1cs_bin_format.md>
/// - <https://github.com/iden3/snarkjs/blob/master/src/r1cs_export_json.js>
#[derive(Debug, Clone, Deserialize)]
pub struct CircomR1CS {
    /// Number of bytes per field element.
    pub n8: usize,
    /// Order of the field used in this R1CS.
    pub prime: String,
    /// Number of variables in total.
    #[serde(rename = "nVars")]
    pub num_vars: usize,
    /// Number of outputs (public).
    #[serde(rename = "nOutputs")]
    pub num_outputs: usize,
    /// Number of public inputs, does not include the constant term!
    #[serde(rename = "nPubInputs")]
    pub num_pub_inputs: usize,
    /// Number of private inputs.
    #[serde(rename = "nPrvInputs")]
    pub num_priv_inputs: usize,
    /// Number of labels.
    #[serde(rename = "nLabels")]
    pub num_labels: usize,
    /// Number of constraints.
    #[serde(rename = "nConstraints")]
    pub num_constraints: usize,
    /// Constraints, where each `HashMap` contains mapping from witness
    /// index to coefficient for that linear combination.
    #[serde(deserialize_with = "constraint_deserializer")]
    pub constraints: Vec<[HashMap<usize, FrElement>; 3]>,
}

fn constraint_deserializer<'de, D>(
    deserializer: D,
) -> Result<Vec<[HashMap<usize, FrElement>; 3]>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw_constraints: Vec<[HashMap<usize, String>; 3]> = Deserialize::deserialize(deserializer)?;

    Ok(raw_constraints
        .into_iter()
        .map(|[a, b, c]| {
            [
                a.into_iter()
                    .map(|(k, v)| (k, circom_str_to_lambda_field_element(v)))
                    .collect(),
                b.into_iter()
                    .map(|(k, v)| (k, circom_str_to_lambda_field_element(v)))
                    .collect(),
                c.into_iter()
                    .map(|(k, v)| (k, circom_str_to_lambda_field_element(v)))
                    .collect(),
            ]
        })
        .collect())
}

/// Reads and parses an R1CS file at the given path.
#[inline]
pub fn read_circom_r1cs(path: impl AsRef<Path>) -> Result<CircomR1CS, CircomReaderError> {
    let file = File::open(path).map_err(CircomReaderError::IOError)?;
    serde_json::from_reader(BufReader::new(file)).map_err(CircomReaderError::SerdeError)
}

/// Reads and parses a witness file at the given path.
#[inline]
pub fn read_circom_witness(path: impl AsRef<Path>) -> Result<CircomWitness, CircomReaderError> {
    let file = File::open(path).map_err(CircomReaderError::IOError)?;
    let wtns: Vec<String> =
        serde_json::from_reader(BufReader::new(file)).map_err(CircomReaderError::SerdeError)?;

    Ok(wtns
        .into_iter()
        .map(circom_str_to_lambda_field_element)
        .collect())
}

/// Converts a string Circom field element to Lambda field element.
///
/// The Circom field element can be in either decimal or hexadecimal format;
/// otherwise, this function will panic.
#[inline]
fn circom_str_to_lambda_field_element(value: impl AsRef<str>) -> FrElement {
    let value = value.as_ref();

    if let Ok(big_uint) = UnsignedInteger::<4>::from_dec_str(value) {
        FrElement::from(&big_uint)
    } else if let Ok(big_uint) = UnsignedInteger::<4>::from_hex(value) {
        FrElement::from(&big_uint)
    } else {
        panic!("Could not parse field element from string: {value}");
    }
}
