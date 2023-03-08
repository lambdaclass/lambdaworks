use lambdaworks_crypto::{commitments::kzg::{StructuredReferenceString, KateZaveruchaGoldberg}, fiat_shamir::transcript::{self, Transcript}};
use lambdaworks_math::{elliptic_curve::{short_weierstrass::curves::bls12_381::{pairing::BLS12381AtePairing}, traits::IsPairing}, field::{element::FieldElement, fields::montgomery_backed_prime_fields::{U256PrimeField, IsMontgomeryConfiguration, MontgomeryBackendPrimeField}}, unsigned_integer::element::{UnsignedInteger, U256}};
use lambdaworks_math::traits::ByteConversion;
use crate::{setup::{VerificationKey, Circuit}, config::{G1Point, FrElement, MAXIMUM_DEGREE, KZG, G2Point}};


struct Proof {
    // Round 1
    a_1: G1Point, // [a(x)]₁ (commitment to left wire polynomial)
    b_1: G1Point, // [b(x)]₁ (commitment to right wire polynomial)
    c_1: G1Point, // [c(x)]₁ (commitment to output wire polynomial)

    // Round 2
    z_1: G1Point, // [z(x)]₁ (commitment to permutation polynomial)

    // Round 3
    t_lo_1: G1Point, // [t_lo(x)]₁ (commitment to t_lo(X), the low chunk of the quotient polynomial t(X))
    t_mid_1: G1Point, // [t_mid(x)]₁ (commitment to t_mid(X), the middle chunk of the quotient polynomial t(X))
    t_hi_1: G1Point, // [t_hi(x)]₁ (commitment to t_hi(X), the high chunk of the quotient polynomial t(X))

    // Round 4
    a_eval: FrElement, // Evaluation of a(X) at evaluation challenge ζ
    b_eval: FrElement, // Evaluation of b(X) at evaluation challenge ζ
    c_eval: FrElement, // Evaluation of c(X) at evaluation challenge ζ
    s1_eval: FrElement, // Evaluation of the first permutation polynomial S_σ1(X) at evaluation challenge ζ
    s2_eval: FrElement, // Evaluation of the second permutation polynomial S_σ2(X) at evaluation challenge ζ
    z_shifted_eval: FrElement, // Evaluation of the shifted permutation polynomial z(X) at the shifted evaluation challenge ζω
    
    // Round 5
    W_z_1: G1Point, // [W_ζ(X)]₁ (commitment to the opening proof polynomial)
    W_zw_1: G1Point, // [W_ζω(X)]₁ (commitment to the opening proof polynomial)
}

fn round_1(
    circuit: Circuit,
    key: VerificationKey<G1Point>,
    srs: StructuredReferenceString<MAXIMUM_DEGREE, G1Point, G2Point>
) -> (G1Point, G1Point, G1Point) {
    let (a_polynomial, b_polynomial, c_polynomial) = circuit.get_program_trace_polynomials();
    let kzg = KZG::new(srs);
    
    let a_1 = kzg.commit(&a_polynomial);
    let b_1 = kzg.commit(&b_polynomial);
    let c_1 = kzg.commit(&c_polynomial);

    (a_1, b_1, c_1)
}

fn prove(
    circuit: Circuit,
    key: VerificationKey<G1Point>,
    srs: StructuredReferenceString<MAXIMUM_DEGREE, G1Point, G2Point>,
) {
    let mut transcript = Transcript::new();

    let (a_1, b_1, c_1) = round_1(circuit, key, srs);
    transcript.append(&a_1.to_bytes_be());
    transcript.append(&b_1.to_bytes_be());
    transcript.append(&c_1.to_bytes_be());
}

#[cfg(test)]
mod tests {
    use super::round_1;

    fn round_1_works() {
    }
}