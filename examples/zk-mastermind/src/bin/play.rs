//! Interactive ZK Mastermind Game
//!
//! Play Mastermind with zero-knowledge proofs!
//!
//! Usage:
//!   cargo run --bin play
//!   cargo run --bin play -- --seed 12345

use std::io::{self, Write};
use std::time::Instant;

use zk_mastermind::{
    compute_secret_commitment, generate_proof, proof_size, verify_proof, Color, GameState, Guess,
    SecretCode,
};

fn main() {
    println!();
    println!("  ╔═══════════════════════════════════════════════════════════╗");
    println!("  ║          ZK MASTERMIND - Interactive Edition              ║");
    println!("  ║       Crack the code with Zero-Knowledge Proofs!          ║");
    println!("  ╚═══════════════════════════════════════════════════════════╝");
    println!();

    // Get seed from command line or prompt user
    let seed = get_seed();
    println!();

    // Generate secret from seed
    let secret = generate_secret_from_seed(seed);
    let game = GameState::new(secret.clone());

    // Compute the commitment (this would be shared publicly before the game starts)
    let commitment = compute_secret_commitment(&secret);

    println!(
        "  A secret 4-color code has been generated from seed: {}",
        seed
    );
    println!(
        "  Secret commitment: {} (prevents cheating)",
        commitment.representative().to_string().chars().take(16).collect::<String>()
    );
    println!();
    print_rules();

    let max_turns = 10;
    let mut turn = 1;

    loop {
        println!();
        println!("  ┌─────────────────────────────────────────────────────────┐");
        println!(
            "  │  Turn {}/{}                                              │",
            turn, max_turns
        );
        println!("  └─────────────────────────────────────────────────────────┘");
        println!();

        // Get guess from user
        let guess = match get_guess() {
            Some(g) => g,
            None => {
                println!("  Goodbye!");
                return;
            }
        };

        // Calculate feedback
        let feedback = game.respond(&guess);

        // Generate and verify ZK proof
        println!();
        print!("  Generating ZK proof... ");
        io::stdout().flush().unwrap();

        let start = Instant::now();
        let proof = match generate_proof(&secret, &guess, &feedback) {
            Ok(p) => p,
            Err(e) => {
                println!("Error: {}", e);
                continue;
            }
        };
        let prove_time = start.elapsed();
        println!("done! ({:?})", prove_time);

        print!("  Verifying proof... ");
        io::stdout().flush().unwrap();

        let start = Instant::now();
        let is_valid = verify_proof(&proof, &guess, &feedback, commitment);
        let verify_time = start.elapsed();

        if is_valid {
            println!("VALID ({:?})", verify_time);
        } else {
            println!("INVALID!");
            println!("  Something went wrong with the proof!");
            continue;
        }

        // Display result
        println!();
        println!("  ┌─────────────────────────────────────────────────────────┐");
        println!(
            "  │  Your guess: {}  {}  {}  {}                               │",
            color_display(guess.0[0]),
            color_display(guess.0[1]),
            color_display(guess.0[2]),
            color_display(guess.0[3])
        );
        println!("  │                                                         │");
        println!(
            "  │  Result: {} exact, {} partial                             │",
            feedback.exact, feedback.partial
        );
        println!(
            "  │  Proof size: {} bytes                                   │",
            proof_size(&proof)
        );
        println!("  └─────────────────────────────────────────────────────────┘");

        // Check for win
        if feedback.is_win() {
            println!();
            println!("  ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★");
            println!("  ★                                                       ★");
            println!(
                "  ★   CONGRATULATIONS! You cracked the code in {} turns!  ★",
                turn
            );
            println!("  ★                                                       ★");
            println!(
                "  ★   The secret was: {}  {}  {}  {}                       ★",
                color_display(secret.0[0]),
                color_display(secret.0[1]),
                color_display(secret.0[2]),
                color_display(secret.0[3])
            );
            println!("  ★                                                       ★");
            println!("  ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★");
            println!();
            return;
        }

        turn += 1;
        if turn > max_turns {
            println!();
            println!("  ╔═══════════════════════════════════════════════════════════╗");
            println!("  ║  GAME OVER - You ran out of turns!                        ║");
            println!("  ║                                                           ║");
            println!(
                "  ║  The secret was: {}  {}  {}  {}                              ║",
                color_display(secret.0[0]),
                color_display(secret.0[1]),
                color_display(secret.0[2]),
                color_display(secret.0[3])
            );
            println!("  ╚═══════════════════════════════════════════════════════════╝");
            println!();
            return;
        }
    }
}

fn get_seed() -> u64 {
    // Check command line args
    let args: Vec<String> = std::env::args().collect();
    for i in 0..args.len() {
        if args[i] == "--seed" && i + 1 < args.len() {
            if let Ok(seed) = args[i + 1].parse::<u64>() {
                return seed;
            }
        }
    }

    // Prompt user
    println!("  Enter a seed number (or press Enter for random):");
    print!("  > ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let input = input.trim();

    if input.is_empty() {
        // Use current time as seed
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        println!("  Using random seed: {}", seed);
        seed
    } else {
        input.parse::<u64>().unwrap_or_else(|_| {
            println!("  Invalid seed, using 42");
            42
        })
    }
}

fn generate_secret_from_seed(seed: u64) -> SecretCode {
    // Simple PRNG based on seed
    let colors = Color::all();
    let mut state = seed;

    let mut code = [Color::Red; 4];
    for item in &mut code {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let idx = ((state >> 33) % 6) as usize;
        *item = colors[idx];
    }

    SecretCode::new(code)
}

fn print_rules() {
    println!("  RULES:");
    println!("  - Guess the secret 4-color code");
    println!("  - After each guess, you'll get feedback:");
    println!("    * EXACT: correct color in correct position");
    println!("    * PARTIAL: correct color in wrong position");
    println!("  - Each guess is verified with a ZK proof!");
    println!();
    println!("  COLORS: R=Red, B=Blue, G=Green, Y=Yellow, O=Orange, P=Purple");
    println!();
    println!("  Enter guesses as 4 letters, e.g.: RBGY or r b g y");
    println!("  Type 'quit' or 'q' to exit");
}

fn get_guess() -> Option<Guess> {
    loop {
        print!("  Enter your guess: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim().to_uppercase();

        if input == "QUIT" || input == "Q" {
            return None;
        }

        // Parse colors from input
        let chars: Vec<char> = input.chars().filter(|c| !c.is_whitespace()).collect();

        if chars.len() != 4 {
            println!("  Please enter exactly 4 colors (e.g., RBGY)");
            continue;
        }

        let mut colors = [Color::Red; 4];
        let mut valid = true;

        for (i, c) in chars.iter().enumerate() {
            match c {
                'R' => colors[i] = Color::Red,
                'B' => colors[i] = Color::Blue,
                'G' => colors[i] = Color::Green,
                'Y' => colors[i] = Color::Yellow,
                'O' => colors[i] = Color::Orange,
                'P' => colors[i] = Color::Purple,
                _ => {
                    println!("  Unknown color '{}'. Use R, B, G, Y, O, or P", c);
                    valid = false;
                    break;
                }
            }
        }

        if valid {
            return Some(Guess::new(colors));
        }
    }
}

fn color_display(color: Color) -> &'static str {
    match color {
        Color::Red => "R",
        Color::Blue => "B",
        Color::Green => "G",
        Color::Yellow => "Y",
        Color::Orange => "O",
        Color::Purple => "P",
    }
}
