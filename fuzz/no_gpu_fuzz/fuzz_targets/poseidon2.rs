#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use lambdaworks_crypto::hash::poseidon2::{Fp, Poseidon2};
use libfuzzer_sys::fuzz_target;

/// Goldilocks prime: p = 2^64 - 2^32 + 1
const GOLDILOCKS_PRIME: u64 = 0xFFFF_FFFF_0000_0001;

/// Generate a field element with bias toward interesting boundary values.
/// This covers:
/// - Normal random values (most common)
/// - Values near 0
/// - Values near the prime (p-1, p-2, etc.)
/// - Values near 2^32 (the "hole" in Goldilocks)
fn arbitrary_field_element(u: &mut Unstructured) -> arbitrary::Result<Fp> {
    let choice: u8 = u.int_in_range(0..=9)?;
    let val = match choice {
        // 60% normal random u64 (will be reduced mod p)
        0..=5 => u.arbitrary::<u64>()?,
        // 10% near zero
        6 => u.int_in_range(0..=1000)?,
        // 10% near prime (these reduce to small values or wrap)
        7 => GOLDILOCKS_PRIME.wrapping_sub(u.int_in_range(0..=1000)?),
        // 10% near 2^32 boundary
        8 => {
            let offset: u64 = u.int_in_range(0..=1000)?;
            (1u64 << 32).wrapping_add(offset.wrapping_sub(500))
        }
        // 10% exact boundary values
        _ => *u.choose(&[0, 1, GOLDILOCKS_PRIME - 1, 1 << 32, (1 << 32) - 1])?,
    };
    Ok(Fp::from(val))
}

#[derive(Debug)]
struct FuzzInput {
    // Use Vec to test arbitrary lengths:
    // - Empty (edge case)
    // - < RATE (single permutation)
    // - > RATE (multi-round absorption)
    inputs: Vec<Fp>,
    // Dedicated inputs for specific property tests
    a: Fp,
    b: Fp,
    c: Fp,
    d: Fp,
}

impl<'a> Arbitrary<'a> for FuzzInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let len: usize = u.int_in_range(0..=20)?;
        let mut inputs = Vec::with_capacity(len);
        for _ in 0..len {
            inputs.push(arbitrary_field_element(u)?);
        }
        Ok(FuzzInput {
            inputs,
            a: arbitrary_field_element(u)?,
            b: arbitrary_field_element(u)?,
            c: arbitrary_field_element(u)?,
            d: arbitrary_field_element(u)?,
        })
    }
}

fuzz_target!(|fuzz_data: FuzzInput| {
    let elements = &fuzz_data.inputs;
    let a = fuzz_data.a;
    let b = fuzz_data.b;
    let c = fuzz_data.c;
    let d = fuzz_data.d;

    // 1. Variable Length & Consistency check
    // hash_vec should be consistent with hash_single (len=1) and hash_many (len!=1)
    // Note: hash_vec panics on empty input, so we only test non-empty cases
    if !elements.is_empty() {
        let vec_hash = Poseidon2::hash_vec(elements);

        if elements.len() == 1 {
            assert_eq!(
                vec_hash,
                Poseidon2::hash_single(&elements[0]),
                "hash_vec(len=1) != hash_single"
            );
        } else {
            assert_eq!(
                vec_hash,
                Poseidon2::hash_many(elements),
                "hash_vec(len!=1) != hash_many"
            );
        }
    }

    // Verify hash_many handles empty input (applies padding, returns non-zero)
    let many_hash_empty = Poseidon2::hash_many(&[]);
    assert_ne!(
        many_hash_empty,
        [Fp::zero(); 2],
        "hash_many([]) should be non-zero due to padding"
    );

    // 2. Domain separation: hash(a,b) != hash_many([a,b])
    // hash uses domain tag 2, hash_many uses 10* padding
    assert_ne!(
        Poseidon2::hash(&a, &b),
        Poseidon2::hash_many(&[a, b]),
        "Domain separation violated: hash(a,b) == hash_many([a,b])"
    );

    // 3. Non-commutativity for compress (when inputs differ)
    let left = [a, b];
    let right = [c, d];
    if left != right {
        assert_ne!(
            Poseidon2::compress(&left, &right),
            Poseidon2::compress(&right, &left),
            "Compress should be non-commutative"
        );
    }

    // 4. Determinism
    assert_eq!(
        Poseidon2::hash(&a, &b),
        Poseidon2::hash(&a, &b),
        "Hash should be deterministic"
    );
    assert_eq!(
        Poseidon2::hash_single(&a),
        Poseidon2::hash_single(&a),
        "hash_single should be deterministic"
    );
    assert_eq!(
        Poseidon2::compress(&left, &right),
        Poseidon2::compress(&left, &right),
        "compress should be deterministic"
    );

    // 5. Domain separation: hash_single vs hash_many
    assert_ne!(
        Poseidon2::hash_single(&a),
        Poseidon2::hash_many(&[a]),
        "hash_single should differ from hash_many"
    );

    // 6. Length extension resistance
    assert_ne!(
        Poseidon2::hash_many(&[a, b]),
        Poseidon2::hash_many(&[a, b, c]),
        "Different length inputs should produce different hashes"
    );

    // 7. Prefix resistance
    assert_ne!(
        Poseidon2::hash_many(&[a, b, c]),
        Poseidon2::hash_many(&[b, c]),
        "Prefix removal should change hash"
    );

    // 8. Non-zero outputs
    assert_ne!(
        Poseidon2::hash(&a, &b),
        [Fp::zero(); 2],
        "Hash should not be [0, 0]"
    );
    assert_ne!(
        Poseidon2::hash_single(&a),
        [Fp::zero(); 2],
        "Hash single should not be [0, 0]"
    );
    assert_ne!(
        Poseidon2::compress(&left, &right),
        [Fp::zero(); 2],
        "Compress should not be [0, 0]"
    );

    // 9. Collision resistance
    if a != b {
        assert_ne!(
            Poseidon2::hash_single(&a),
            Poseidon2::hash_single(&b),
            "hash_single should be collision-resistant"
        );
    }

    // 10. Domain separation: hash vs compress
    // hash(a, b) uses domain tag = 2
    // compress([a, b], [0, 0]) uses domain tag = 4
    let left_for_sep = [a, b];
    let right_for_sep = [Fp::zero(), Fp::zero()];
    assert_ne!(
        Poseidon2::hash(&a, &b),
        Poseidon2::compress(&left_for_sep, &right_for_sep),
        "hash and compress should have domain separation"
    );
});
