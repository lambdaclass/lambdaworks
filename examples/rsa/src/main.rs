mod rsa;

use lambdaworks_math::unsigned_integer::element::UnsignedInteger;
use rsa::RSA;
use std::error::Error;

/// Demonstrates basic RSA using small numbers (for educational purposes only).
fn demo_basic_rsa() -> Result<(), Box<dyn Error>> {
    println!("\nExample 1: Basic RSA Concepts");
    println!("============================");
    println!("This example uses small prime numbers to demonstrate RSA mechanics.");
    println!("WARNING: Small primes are NOT secure for real use!\n");

    // Choose small prime numbers
    let p = UnsignedInteger::from_u64(61);
    let q = UnsignedInteger::from_u64(53);
    println!("Selected primes:");
    println!("p = {}", p);
    println!("q = {}", q);
    // Create RSA instance
    let rsa = RSA::new(p, q).unwrap();

    // Get public and private keys
    let (e, n) = rsa.public_key();
    let d = rsa.secret_key();
    println!("\nRSA Parameters:");
    println!("n (modulus) = p * q = {}", n);
    println!("e (public exponent) = {}", e);
    println!("d (private exponent) = {}", d);

    // Encrypt and decrypt a small number
    let message = UnsignedInteger::from_u64(42);
    println!("\nExample A: Numeric Encryption");
    println!("Message: {}", message);

    let ciphertext = rsa.encrypt(&message)?;
    println!("Ciphertext: {}", ciphertext);

    let decrypted = rsa.decrypt(&ciphertext)?;
    println!("Decrypted: {}", decrypted);
    println!("Message recovered: {}", decrypted == message);

    Ok(())
}

/// Demonstrates simple text encryption using RSA (without padding)
fn demo_text_encryption() -> Result<(), Box<dyn Error>> {
    println!("\nExample 2: Simple Text Encryption");
    println!("==============================");
    println!("This example shows how to encrypt text using basic RSA.");
    println!("Note: This is a simplified version without padding.\n");

    let p = UnsignedInteger::from_u64(61);
    let q = UnsignedInteger::from_u64(53);
    let rsa = RSA::new(p, q).unwrap();

    // Encrypt a single character
    let message = b"A"; // ASCII 'A' = 65
    println!(
        "Original character: '{}' (ASCII value: {})",
        String::from_utf8_lossy(message),
        message[0]
    );

    let cipher = rsa.encrypt_bytes_simple(message)?;
    println!("Encrypted (hex): {}", hex::encode(&cipher));

    let recovered = rsa.decrypt_bytes_simple(&cipher)?;
    println!("Decrypted: '{}'", String::from_utf8_lossy(&recovered));
    println!("Message recovered: {}", message == &recovered[..]);

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("RSA Encryption Examples");
    println!("======================");
    println!("These examples demonstrate the core concepts of RSA encryption.");
    println!("For simplicity, we use small prime numbers and basic operations.");
    println!("In practice, you would use:");
    println!("- Large prime numbers (2048+ bits)");
    println!("- Proper padding schemes (e.g., PKCS#1)");
    println!("- Secure random number generation");

    demo_basic_rsa()?;
    demo_text_encryption()?;

    // Additional example with known RSA parameters
    let e = UnsignedInteger::from_u64(17);
    let d = UnsignedInteger::from_u64(2753);
    let n = UnsignedInteger::from_u64(3233);
    let rsa = RSA { e, d, n };

    let message = UnsignedInteger::from_u64(42);
    println!("\nAdditional Example:");
    println!("Original message: {}", message);

    let ciphertext = rsa.encrypt(&message)?;
    println!("Encrypted: {}", ciphertext);

    let decrypted = rsa.decrypt(&ciphertext)?;
    println!("Decrypted: {}", decrypted);

    if message == decrypted {
        println!("Successfully verified encryption/decryption!");
    } else {
        println!("Error: Decrypted message does not match.");
    }

    Ok(())
}
