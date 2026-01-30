//! ZK Mastermind Demo
//!
//! This demo shows a complete game of Mastermind with ZK proofs.
//!
//! Usage:
//!   cargo run

use std::time::Instant;

use zk_mastermind::{
    generate_proof, proof_size, verify_proof, Color, GameState, Guess, SecretCode,
};

fn main() {
    println!("==================================================================");
    println!("            ZK Mastermind - STARK Prover Demo");
    println!("        Zero-Knowledge Proof of Mastermind Feedback");
    println!("==================================================================");
    println!();

    // Legend
    println!("Color Legend:");
    println!("  R = Red, B = Blue, G = Green, Y = Yellow, O = Orange, P = Purple");
    println!();

    // 1. Setup: CodeMaker chooses a secret
    let secret = SecretCode::new([Color::Red, Color::Blue, Color::Green, Color::Yellow]);
    let game = GameState::new(secret.clone());

    println!("Setup Phase");
    println!("   CodeMaker has chosen a secret code: [R, B, G, Y]");
    println!("   Secret commitment (simulated hash): 0x1234...5678");
    println!();

    // 2. Turn 1: CodeBreaker makes a guess
    println!("------------------------------------------------------------------");
    println!("Turn 1");
    println!("------------------------------------------------------------------");

    let guess1 = Guess::new([Color::Red, Color::Green, Color::Blue, Color::Yellow]);
    println!("   CodeBreaker guesses: {}", guess1);

    // CodeMaker calculates feedback and generates proof
    let start = Instant::now();
    let feedback1 = game.respond(&guess1);
    println!("   CodeMaker calculates feedback: {}", feedback1);

    // Generate ZK proof
    let proof1 =
        generate_proof(&secret, &guess1, &feedback1).expect("Proof generation should succeed");
    let prove_time1 = start.elapsed();

    println!("   Proof generation time: {:?}", prove_time1);
    println!("   Proof size: {} bytes", proof_size(&proof1));

    // CodeBreaker verifies the proof
    let start = Instant::now();
    let is_valid1 = verify_proof(&proof1, &guess1, &feedback1);
    let verify_time1 = start.elapsed();

    println!("   Verification time: {:?}", verify_time1);
    if is_valid1 {
        println!("   Proof verified! Feedback is correct.");
    } else {
        println!("   Proof verification failed!");
    }
    println!();

    // 3. Turn 2: CodeBreaker makes another guess
    println!("------------------------------------------------------------------");
    println!("Turn 2");
    println!("------------------------------------------------------------------");

    let guess2 = Guess::new([Color::Red, Color::Blue, Color::Green, Color::Purple]);
    println!("   CodeBreaker guesses: {}", guess2);

    let start = Instant::now();
    let feedback2 = game.respond(&guess2);
    println!("   CodeMaker calculates feedback: {}", feedback2);

    let proof2 =
        generate_proof(&secret, &guess2, &feedback2).expect("Proof generation should succeed");
    let prove_time2 = start.elapsed();

    println!("   Proof generation time: {:?}", prove_time2);
    println!("   Proof size: {} bytes", proof_size(&proof2));

    let start = Instant::now();
    let is_valid2 = verify_proof(&proof2, &guess2, &feedback2);
    let verify_time2 = start.elapsed();

    println!("   Verification time: {:?}", verify_time2);
    if is_valid2 {
        println!("   Proof verified! Feedback is correct.");
    } else {
        println!("   Proof verification failed!");
    }
    println!();

    // 4. Turn 3: CodeBreaker solves it
    println!("------------------------------------------------------------------");
    println!("Turn 3 (Final)");
    println!("------------------------------------------------------------------");

    let guess3 = Guess::new([Color::Red, Color::Blue, Color::Green, Color::Yellow]);
    println!("   CodeBreaker guesses: {}", guess3);

    let start = Instant::now();
    let feedback3 = game.respond(&guess3);
    println!("   CodeMaker calculates feedback: {}", feedback3);

    let proof3 =
        generate_proof(&secret, &guess3, &feedback3).expect("Proof generation should succeed");
    let prove_time3 = start.elapsed();

    println!("   Proof generation time: {:?}", prove_time3);
    println!("   Proof size: {} bytes", proof_size(&proof3));

    let start = Instant::now();
    let is_valid3 = verify_proof(&proof3, &guess3, &feedback3);
    let verify_time3 = start.elapsed();

    println!("   Verification time: {:?}", verify_time3);
    if is_valid3 {
        println!("   Proof verified! Feedback is correct.");
    } else {
        println!("   Proof verification failed!");
    }

    if feedback3.is_win() {
        println!();
        println!("   CodeBreaker wins! Code broken!");
    }
    println!();

    // 5. Summary
    println!("==================================================================");
    println!("                       Statistics");
    println!("==================================================================");
    println!();
    println!("Proof Generation:");
    println!("  Turn 1: {:?}", prove_time1);
    println!("  Turn 2: {:?}", prove_time2);
    println!("  Turn 3: {:?}", prove_time3);
    println!(
        "  Average: {:?}",
        (prove_time1 + prove_time2 + prove_time3) / 3
    );
    println!();
    println!("Proof Verification:");
    println!("  Turn 1: {:?}", verify_time1);
    println!("  Turn 2: {:?}", verify_time2);
    println!("  Turn 3: {:?}", verify_time3);
    println!(
        "  Average: {:?}",
        (verify_time1 + verify_time2 + verify_time3) / 3
    );
    println!();
    println!("Proof Size: ~{} bytes per proof", proof_size(&proof1));
    println!();
    println!("Field: Stark252PrimeField (252-bit prime field)");
    println!("Proof System: STARK (Scalable Transparent Arguments of Knowledge)");
    println!();
    println!("Demo completed successfully!");
}
