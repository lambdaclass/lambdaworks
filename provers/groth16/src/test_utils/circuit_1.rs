use crate::qap::QAP;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement;

/// Builds QAP representation for equation x^3 + x + 5 = 35, based on Vitalik's post
///https://vitalik.ca/general/2016/12/10/qap.html
pub fn qap_example_circuit_1() -> QAP {
    QAP::new(
        1,
        // TODO: Roots of unity
        ["0x1", "0x2", "0x3", "0x4"]
            .map(|e| FrElement::from_hex_unchecked(e))
            .to_vec(),
        [
            ["0", "0", "0", "5"],
            ["1", "0", "1", "0"],
            ["0", "0", "0", "0"],
            ["0", "1", "0", "0"],
            ["0", "0", "1", "0"],
            ["0", "0", "0", "1"],
        ]
        .map(|col| col.to_vec())
        .to_vec(),
        [
            ["0", "0", "1", "1"],
            ["1", "1", "0", "0"],
            ["0", "0", "0", "0"],
            ["0", "0", "0", "0"],
            ["0", "0", "0", "0"],
            ["0", "0", "0", "0"],
        ]
        .map(|col| col.to_vec())
        .to_vec(),
        [
            ["0", "0", "0", "0"],
            ["0", "0", "0", "0"],
            ["0", "0", "0", "1"],
            ["1", "0", "0", "0"],
            ["0", "1", "0", "0"],
            ["0", "0", "1", "0"],
        ]
        .map(|col| col.to_vec())
        .to_vec(),
    )
}
