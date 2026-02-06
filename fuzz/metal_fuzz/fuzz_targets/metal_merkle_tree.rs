//! Differential fuzzer for Metal GPU Poseidon Merkle tree vs CPU.
//!
//! Tests three operations with each fuzzed input:
//! 1. `hash_leaves_gpu` vs CPU `hash_single` (leaf hashing)
//! 2. `hash_level_gpu` vs CPU `hash` (pair hashing)
//! 3. Full `MerkleTree::build` root comparison

#![no_main]

use libfuzzer_sys::fuzz_target;

use lambdaworks_crypto::{
    hash::poseidon::{starknet::PoseidonCairoStark252, Poseidon},
    merkle_tree::{
        backends::{field_element::TreePoseidon, metal::MetalPoseidonBackend},
        merkle::MerkleTree,
    },
};
use lambdaworks_math::{
    field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    },
    unsigned_integer::element::U256,
};

type FE = FieldElement<Stark252PrimeField>;

fuzz_target!(|data: Vec<[u64; 4]>| {
    if data.is_empty() {
        return;
    }

    let leaves: Vec<FE> = data
        .iter()
        .map(|limbs| FE::from(&U256 { limbs: *limbs }))
        .collect();

    // --- 1. hash_leaves_gpu vs CPU hash_single ---
    // Pad to >= 64 elements to exercise the GPU path
    let mut padded = leaves.clone();
    while padded.len() < 64 {
        padded.push(FE::from(0u64));
    }

    let gpu_leaves =
        MetalPoseidonBackend::hash_leaves_gpu(&padded).expect("hash_leaves_gpu failed");
    let cpu_leaves: Vec<FE> = padded
        .iter()
        .map(PoseidonCairoStark252::hash_single)
        .collect();

    for (i, (gpu, cpu)) in gpu_leaves.iter().zip(cpu_leaves.iter()).enumerate() {
        assert_eq!(
            gpu, cpu,
            "hash_single mismatch at index {}: GPU={:?} CPU={:?}",
            i, gpu, cpu
        );
    }

    // --- 2. hash_level_gpu vs CPU hash ---
    // Use the hashed leaves as input nodes; ensure even count and >= 64
    let mut nodes = cpu_leaves;
    if nodes.len() % 2 != 0 {
        nodes.push(FE::from(0u64));
    }
    while nodes.len() < 64 {
        nodes.push(FE::from(0u64));
        nodes.push(FE::from(0u64));
    }

    let gpu_level = MetalPoseidonBackend::hash_level_gpu(&nodes).expect("hash_level_gpu failed");
    let output_count = nodes.len() / 2;
    let cpu_level: Vec<FE> = (0..output_count)
        .map(|i| PoseidonCairoStark252::hash(&nodes[i * 2], &nodes[i * 2 + 1]))
        .collect();

    for (i, (gpu, cpu)) in gpu_level.iter().zip(cpu_level.iter()).enumerate() {
        assert_eq!(
            gpu, cpu,
            "hash_level mismatch at pair {}: GPU={:?} CPU={:?}",
            i, gpu, cpu
        );
    }

    // --- 3. Full MerkleTree build: compare roots ---
    let metal_tree = MerkleTree::<MetalPoseidonBackend>::build(&leaves);
    let cpu_tree = MerkleTree::<TreePoseidon<PoseidonCairoStark252>>::build(&leaves);

    match (metal_tree, cpu_tree) {
        (Some(metal), Some(cpu)) => {
            assert_eq!(
                metal.root,
                cpu.root,
                "Merkle root mismatch for {} leaves",
                leaves.len()
            );
        }
        (None, None) => {}
        (metal, cpu) => {
            panic!(
                "Build disagreement for {} leaves: Metal={} CPU={}",
                leaves.len(),
                metal.is_some(),
                cpu.is_some()
            );
        }
    }
});
