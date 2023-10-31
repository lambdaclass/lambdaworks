use crate::{common::*, qap::QAP};

/// Builds QAP representation for equation x^3 + x + 5 = 35, based on Vitalik's post
///https://vitalik.ca/general/2016/12/10/qap.html
pub fn qap_example_circuit_1() -> QAP {
    let gate_indices = ["0x1", "0x2", "0x3", "0x4"];
    let num_of_public_inputs = 1;
    let [l, r, o] = [
        [
            ["0", "0", "0", "5"],
            ["1", "0", "1", "0"],
            ["0", "0", "0", "0"],
            ["0", "1", "0", "0"],
            ["0", "0", "1", "0"],
            ["0", "0", "0", "1"],
        ],
        [
            ["0", "0", "1", "1"],
            ["1", "1", "0", "0"],
            ["0", "0", "0", "0"],
            ["0", "0", "0", "0"],
            ["0", "0", "0", "0"],
            ["0", "0", "0", "0"],
        ],
        [
            ["0", "0", "0", "0"],
            ["0", "0", "0", "0"],
            ["0", "0", "0", "1"],
            ["1", "0", "0", "0"],
            ["0", "1", "0", "0"],
            ["0", "0", "1", "0"],
        ],
    ]
    .map(|matrix| matrix.map(|col| col.to_vec()).to_vec());

    QAP::from_hex_matrices(
        num_of_public_inputs,
        gate_indices
            .map(|e| FrElement::from_hex_unchecked(e))
            .to_vec(),
        l,
        r,
        o,
    )
}
