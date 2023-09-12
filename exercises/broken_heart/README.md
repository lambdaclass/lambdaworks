# Loki’s broken heart 

After successfully breaking into Loki’s vault and getting access to some of his finest treasures and weapons, you spot a small trapdoor under a carpet. The trapdoor is locked and contains a device with a PLONK prover. It says: “Prove that the point $(1,y)$ belongs to the elliptic curve $y^2 = x^3 + 4$”. You see that, in order to prove this, you need that $y^2 - x^3 - 4$ is equal to zero, which corresponds to the circuit for the prover provided by Loki. Can you open the trapdoor?

# Description

This challenge is about exploiting a vulnerability in weak Fiat-Shamir implementations.

The idea is to have a small server with an endpoint accepting proofs of executions of circuits. The plonk backend will have a bug in the initialization of the transcript and won't add the public inputs to the transcript. So as long as one public input is in control of the attacker, he can forge fake proofs.

At the moment the circuit is:

```
PUBLIC INPUT: x
PUBLIC INPUT: y

ASSERT 0 == y^2 - x^3 - 4
```
And it instantiated over the `BLS12 381` scalar field.
If the user achieves to send a proof for `x==1`, then they obtain the flag. Since $5$ is a quadratic non residue in the base field of the circuit, this can only be achieved by forging a fake proof.

The vulnerability stems from a bug in the implementation of strong Fiat-Shamir. A correct implementation should add, among other things, all the public inputs to the transcript at initialization. If a public input is not added to the transcript and is in control of the attacker, they can forge a fake proof. Here, fixing `x=1` leaves `y` under control of the user.

The attack is described in Section V of [Weak Fiat-Shamir Attacks on Modern Proof Systems](https://eprint.iacr.org/2023/691.pdf).

Here is a description of the attack.

![image](https://github.com/lambdaclass/challenges-ctf/assets/41742639/d2040ccd-17ad-4f0e-b910-a17ceda96ed4)

Instead of taking random polynomials (steps (1) to (7)), the current solution takes a valid proof for the pair `x=0`, `y=2` and uses it to forge a `y'` for `x=1` that's compatible with the original proof.

At the moment, the server endpoint is simulated with the following function.

```rust
pub fn server_endpoint_verify(
    srs: ChallengeSRS,
    common_preprocessed_input: CommonPreprocessedInput<FrField>,
    vk: &ChallengeVK,
    x: &FrElement,
    y: &FrElement,
    proof: &ChallengeProof,
) -> String {
    let public_input = [x.clone(), y.clone()];
    let kzg = KZG::new(srs);
    let verifier = Verifier::new(kzg);
    let result = verifier.verify(proof, &public_input, &common_preprocessed_input, vk);
    if !result {
        "Invalid Proof".to_string()
    } else if x != &FieldElement::one() {
        "Valid Proof. Congrats!".to_string()
    } else {
        FLAG.to_string()
    }
}
```

The attack can be found in `src/solution.rs` along with a test that showcases it.

## Get it to work

Currently `lambdaworks_plonk_prover` does not expose the weak Fiat-Shamir vulnerability.
So to make the challenge work we need to modify it.

1. Clone `lambdaworks_plonk_prover` repo: `git clone git@github.com:lambdaclass/lambdaworks_plonk_prover.git`
1. `git checkout 07e36bf`
1. Make the following changes to it:

```diff
diff --git a/Cargo.toml b/Cargo.toml
index 7f0e324..c36a00d 100644
--- a/Cargo.toml
+++ b/Cargo.toml
@@ -7,8 +7,8 @@ edition = "2021"
 
 [dependencies]
 serde = { version = "1.0", features = ["derive"]} 
-lambdaworks-math = { git = "https://github.com/lambdaclass/lambdaworks", rev = "943963c" }
-lambdaworks-crypto = { git = "https://github.com/lambdaclass/lambdaworks", rev = "943963c" }
+lambdaworks-math = { git = "https://github.com/lambdaclass/lambdaworks", rev = "d8f14cb" }
+lambdaworks-crypto = { git = "https://github.com/lambdaclass/lambdaworks", rev = "d8f14cb" }
 
 thiserror = "1.0.38"
 serde_json = "1.0"
diff --git a/src/setup.rs b/src/setup.rs
index 493278a..437bcc9 100644
--- a/src/setup.rs
+++ b/src/setup.rs
@@ -69,7 +69,7 @@ pub fn setup<F: IsField, CS: IsCommitmentScheme<F>>(
 
 pub fn new_strong_fiat_shamir_transcript<F, CS>(
     vk: &VerificationKey<CS::Commitment>,
-    public_input: &[FieldElement<F>],
+    _public_input: &[FieldElement<F>],
 ) -> DefaultTranscript
 where
     F: IsField,
@@ -88,9 +88,6 @@ where
     transcript.append(&vk.qo_1.serialize());
     transcript.append(&vk.qc_1.serialize());
 
-    for value in public_input.iter() {
-        transcript.append(&value.to_bytes_be());
-    }
     transcript
 }
```

1. Clone this repo and modify its `Cargo.toml` to point the `lambdaworks-plonk` dependency to your local copy of `lambdaworks_plonk_prover`.
1. Run `cargo test`
