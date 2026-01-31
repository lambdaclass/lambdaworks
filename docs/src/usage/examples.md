# Examples

This page provides practical examples demonstrating common use cases for lambdaworks.

## Field Arithmetic Examples

### Example 1: Basic Field Operations

```rust
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

type FE = FieldElement<Stark252PrimeField>;

fn field_arithmetic_demo() {
    // Creating elements
    let a = FE::from(100u64);
    let b = FE::from(7u64);

    // All arithmetic operations
    let sum = &a + &b;           // 107
    let diff = &a - &b;          // 93
    let prod = &a * &b;          // 700
    let quot = &a / &b;          // 100 * 7^(-1) mod p
    let neg = -&a;               // p - 100
    let squared = a.square();    // 10000
    let cubed = a.pow(3u64);     // 1000000

    // Inverse
    let b_inv = b.inv().expect("non-zero");
    assert_eq!(&b * &b_inv, FE::one());

    // Working with hex
    let c = FE::from_hex_unchecked("0x1234abcd");
    let hex_str = c.representative().to_hex();

    // Batch inversion (efficient for multiple inversions)
    let mut elements = vec![FE::from(2), FE::from(3), FE::from(5)];
    FieldElement::inplace_batch_inverse(&mut elements).expect("all non-zero");
}
```

### Example 2: Polynomial Interpolation for Secret Sharing

Shamir's Secret Sharing uses polynomial interpolation:

```rust
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

type F = U64PrimeField<65537>;
type FE = FieldElement<F>;

fn shamir_secret_sharing() {
    // Secret to share
    let secret = FE::from(12345);

    // Create random polynomial of degree 2 (need 3 shares to reconstruct)
    // p(x) = secret + a1*x + a2*x^2
    let a1 = FE::from(42);  // In practice, use random values
    let a2 = FE::from(17);
    let poly = Polynomial::new(&[secret, a1, a2]);

    // Generate 5 shares (more than threshold)
    let shares: Vec<(FE, FE)> = (1..=5)
        .map(|i| {
            let x = FE::from(i as u64);
            let y = poly.evaluate(&x);
            (x, y)
        })
        .collect();

    println!("Generated {} shares", shares.len());

    // Reconstruct from any 3 shares
    let chosen_shares = &shares[0..3];
    let xs: Vec<FE> = chosen_shares.iter().map(|(x, _)| x.clone()).collect();
    let ys: Vec<FE> = chosen_shares.iter().map(|(_, y)| y.clone()).collect();

    let reconstructed = Polynomial::interpolate(&xs, &ys).expect("interpolation");
    let recovered_secret = reconstructed.evaluate(&FE::zero());

    assert_eq!(recovered_secret, secret);
    println!("Secret recovered: {}", recovered_secret.representative());
}
```

## Elliptic Curve Examples

### Example 3: ECDH Key Exchange

```rust
use lambdaworks_math::elliptic_curve::{
    short_weierstrass::curves::bls12_381::curve::BLS12381Curve,
    traits::IsEllipticCurve,
};
use lambdaworks_math::cyclic_group::IsGroup;

fn ecdh_key_exchange() {
    let g = BLS12381Curve::generator();

    // Alice's keys
    let alice_private = 12345u64;  // In practice, use secure random
    let alice_public = g.operate_with_self(alice_private);

    // Bob's keys
    let bob_private = 67890u64;
    let bob_public = g.operate_with_self(bob_private);

    // Alice computes shared secret
    let alice_shared = bob_public.operate_with_self(alice_private);

    // Bob computes shared secret
    let bob_shared = alice_public.operate_with_self(bob_private);

    // Both arrive at the same point
    assert_eq!(alice_shared.to_affine(), bob_shared.to_affine());
    println!("ECDH key exchange successful!");
    println!("Shared secret (x-coordinate): {:?}",
             &alice_shared.to_affine().x().representative().to_hex()[0..32]);
}
```

### Example 4: Simple Signature Scheme (Educational)

```rust
use lambdaworks_math::elliptic_curve::{
    short_weierstrass::curves::bls12_381::{
        curve::BLS12381Curve,
        default_types::FrElement,
    },
    traits::IsEllipticCurve,
};
use lambdaworks_math::cyclic_group::IsGroup;
use sha3::{Keccak256, Digest};

type FE = FrElement;

fn simple_signature() {
    let g = BLS12381Curve::generator();

    // Key generation
    let private_key = FE::from(123456789u64);  // Use secure random in practice
    let public_key = g.operate_with_self(private_key.representative());

    // Message to sign
    let message = b"Hello, lambdaworks!";

    // Hash message to scalar
    let mut hasher = Keccak256::new();
    hasher.update(message);
    let hash = hasher.finalize();
    let message_scalar = FE::from_bytes_be(&hash[0..32]).expect("valid bytes");

    // Sign: choose random k, compute R = k*G, s = k^(-1) * (hash + private * R.x)
    let k = FE::from(987654321u64);  // Use secure random!
    let r_point = g.operate_with_self(k.representative());
    let r_x = FE::from_bytes_be(&r_point.to_affine().x().to_bytes_be()[0..32])
        .expect("valid");

    let k_inv = k.inv().expect("non-zero");
    let s = &k_inv * &(&message_scalar + &(&private_key * &r_x));

    println!("Message: {:?}", String::from_utf8_lossy(message));
    println!("Signature (r, s) generated");

    // Verify: check that s^(-1) * (hash * G + r_x * PublicKey) = R
    let s_inv = s.inv().expect("non-zero");
    let u1 = &s_inv * &message_scalar;
    let u2 = &s_inv * &r_x;

    let check_point = g.operate_with_self(u1.representative())
        .operate_with(&public_key.operate_with_self(u2.representative()));

    // Compare x-coordinates
    let check_x = FE::from_bytes_be(&check_point.to_affine().x().to_bytes_be()[0..32])
        .expect("valid");

    assert_eq!(check_x, r_x);
    println!("Signature verified!");
}
```

## Merkle Tree Examples

### Example 5: Building and Verifying Merkle Proofs

```rust
use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;
use lambdaworks_crypto::merkle_tree::backends::field_element::FieldElementBackend;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use sha3::Keccak256;

type F = Stark252PrimeField;
type FE = FieldElement<F>;
type Backend = FieldElementBackend<F, Keccak256, 32>;

fn merkle_tree_demo() {
    // Create data (e.g., account balances)
    let balances: Vec<FE> = vec![
        FE::from(1000u64),  // Account 0
        FE::from(2500u64),  // Account 1
        FE::from(500u64),   // Account 2
        FE::from(10000u64), // Account 3
        FE::from(750u64),   // Account 4
        FE::from(3200u64),  // Account 5
        FE::from(100u64),   // Account 6
        FE::from(8888u64),  // Account 7
    ];

    // Build tree
    let tree = MerkleTree::<Backend>::build(&balances)
        .expect("tree construction");

    println!("Merkle tree built with {} leaves", balances.len());
    println!("Root: {:x?}", &tree.root[0..8]);

    // Generate proof for account 3
    let account_index = 3;
    let proof = tree.get_proof_by_pos(account_index)
        .expect("proof generation");

    println!("\nProof for account {}:", account_index);
    println!("  Balance: {}", balances[account_index].representative());
    println!("  Path length: {}", proof.merkle_path.len());

    // Verify - this is what a light client would do
    let is_valid = proof.verify::<Backend>(
        &tree.root,
        account_index,
        &balances[account_index]
    );

    println!("  Verification: {}", if is_valid { "VALID" } else { "INVALID" });

    // Attempt to forge a proof (should fail)
    let fake_balance = FE::from(999999u64);
    let is_fake_valid = proof.verify::<Backend>(
        &tree.root,
        account_index,
        &fake_balance
    );

    println!("  Forged proof verification: {}",
             if is_fake_valid { "VALID (bug!)" } else { "INVALID (correct)" });
}
```

## Polynomial Commitment Examples

### Example 6: KZG Commitment and Opening

```rust
use lambdaworks_crypto::commitments::kzg::{
    KateZaveruchaGoldberg, StructuredReferenceString
};
use lambdaworks_crypto::commitments::traits::IsCommitmentScheme;
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::{
    default_types::{FrElement, FrField},
    pairing::BLS12381AtePairing,
};

type FE = FrElement;
type KZG = KateZaveruchaGoldberg<FrField, BLS12381AtePairing>;

fn kzg_commitment_demo() {
    // Load or generate SRS (simplified for demo)
    let srs = StructuredReferenceString::from_file("test_srs.bin")
        .expect("SRS");
    let kzg = KZG::new(srs);

    // Polynomial: p(x) = 3x^3 + 2x^2 + x + 5
    let p = Polynomial::new(&[
        FE::from(5),  // constant
        FE::from(1),  // x
        FE::from(2),  // x^2
        FE::from(3),  // x^3
    ]);

    println!("Polynomial: p(x) = 3x^3 + 2x^2 + x + 5");

    // Commit
    let commitment = kzg.commit(&p);
    println!("Commitment generated (elliptic curve point)");

    // Prove evaluation at x = 7
    let x = FE::from(7);
    let y = p.evaluate(&x);  // 3*343 + 2*49 + 7 + 5 = 1029 + 98 + 12 = 1139

    println!("\nProving: p(7) = {}", y.representative());

    let proof = kzg.open(&x, &y, &p);
    println!("Opening proof generated");

    // Verify
    let is_valid = kzg.verify(&x, &y, &commitment, &proof);
    println!("Verification: {}", if is_valid { "VALID" } else { "INVALID" });

    // Try to prove wrong evaluation (should fail verification)
    let wrong_y = FE::from(9999);
    let wrong_proof = kzg.open(&x, &wrong_y, &p);  // This will create an invalid proof
    let is_wrong_valid = kzg.verify(&x, &wrong_y, &commitment, &wrong_proof);
    println!("Wrong evaluation verification: {}",
             if is_wrong_valid { "VALID (bug!)" } else { "INVALID (correct)" });
}
```

## STARK Proof Examples

### Example 7: Custom AIR for Squaring

```rust
use stark_platinum_prover::traits::AIR;
use stark_platinum_prover::trace::TraceTable;
use stark_platinum_prover::proof::options::ProofOptions;
use stark_platinum_prover::prover::{IsStarkProver, Prover};
use stark_platinum_prover::verifier::{IsStarkVerifier, Verifier};
use stark_platinum_prover::transcript::StoneProverTranscript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

type F = Stark252PrimeField;
type FE = FieldElement<F>;

// Prove: I know x such that x^8 = result
fn iterated_squaring_proof() {
    // Secret: x = 2
    // Computation: 2 -> 4 -> 16 -> 256 (3 squarings = 2^8)
    let x = FE::from(2);

    // Build trace: each row is the square of the previous
    let mut values = vec![x.clone()];
    for _ in 0..3 {
        let last = values.last().expect("non-empty");
        values.push(last.square());
    }

    // Create trace table
    let trace = TraceTable::<F>::new_from_cols(&[values.clone()]);

    println!("Trace: {:?}",
             values.iter().map(|v| v.representative().to_string()).collect::<Vec<_>>());
    println!("Proving: I know x such that x^8 = 256");

    // Use quadratic AIR example (transition: next = current^2)
    use stark_platinum_prover::examples::quadratic_air::{QuadraticAIR, QuadraticPublicInputs};

    let pub_inputs = QuadraticPublicInputs {
        result: FE::from(256),  // 2^8 = 256
    };

    let proof_options = ProofOptions::default_test_options();
    let transcript = StoneProverTranscript::new(&[]);

    // This would use the quadratic AIR
    // let proof = Prover::<QuadraticAIR<F>>::prove(...);
    // let valid = Verifier::<QuadraticAIR<F>>::verify(...);

    println!("Proof would demonstrate knowledge of 8th root of 256");
}
```

## Circom + Groth16 Example

### Example 8: Proving Circom Circuit

```rust
use lambdaworks_circom_adapter::CircomAdapter;
use lambdaworks_groth16::{setup, Prover, verify};

fn circom_groth16_demo() {
    // Assume we have compiled a Circom circuit:
    // template Multiplier() {
    //     signal input a;
    //     signal input b;
    //     signal output c;
    //     c <== a * b;
    // }

    // Load circuit and witness
    let adapter = CircomAdapter::from_files(
        "multiplier.r1cs",
        "witness.wtns",
    ).expect("loading");

    let (r1cs, witness) = adapter.to_lambdaworks();

    // Setup (in production, from ceremony)
    let (proving_key, verification_key) = setup(&r1cs)
        .expect("setup");

    // Generate proof
    let proof = Prover::prove(&proving_key, &r1cs, &witness)
        .expect("proving");

    // Verify
    let public_inputs = adapter.get_public_inputs();
    let is_valid = verify(&verification_key, &proof, &public_inputs);

    println!("Circom circuit proof: {}",
             if is_valid { "VALID" } else { "INVALID" });
}
```

## Running the Examples

The lambdaworks repository contains complete, runnable examples:

```bash
# Clone the repository
git clone https://github.com/lambdaclass/lambdaworks.git
cd lambdaworks

# Run Shamir secret sharing example
cargo run --example shamir_secret_sharing

# Run Merkle tree CLI
cargo run --example merkle-tree-cli -- --help

# Run Circom integration
cargo run --example prove-verify-circom
```

## More Examples

Additional examples in the repository:

| Example | Description |
|---------|-------------|
| `baby-snark` | Simple SNARK implementation for learning |
| `pinocchio` | First practical SNARK |
| `prove-miden` | Prove Miden VM execution with STARK |
| `schnorr-signature` | Schnorr signature scheme |
| `frost-signature` | FROST threshold signatures |
| `reed-solomon-codes` | Error correction codes |
| `rsa` | Naive RSA implementation |
| `pohlig-hellman-attack` | Educational cryptanalysis |
