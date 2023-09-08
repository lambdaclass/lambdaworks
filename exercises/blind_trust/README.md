# Obi-Wan's search for the Sith Foundry

In his quest to stop the Sith’s menace, Obi-Wan Kenobi finds a (Sith) holocron, giving a zero-knowledge proof of the existence of the Sith’s galactic foundry (using galactic Plonk). This place is rumored to contain several artifacts that could aid the Galactic Republic in its war efforts. The position, given by (x,h,y), satisfies the equation y=x*h+b. After some study, Obi-Wan finds the values of y and b (which belong to Sith lore). The only problem is that, even with this knowledge, it may take him quite long to find the mysterious planet, and the situation in the Republic is desperate. He also finds, together with the holocron, a second item containing the SRS used to generate the proof, the prover, and a description of the circuit used. Will he be able to find the position of the foundry before it is too late? The flag consists of the x and h concatenated and written in hex (for example, x=0x123, h=0x789, the FLAG=123789)

## Description
In this challenge the participants have to exploit a vulnerability in a PLONK implementation that's missing the blindings of the wire polynomials.

The first round of PLONK reads as follows:

```
Compute polynomials a',b',c' as the interpolation polynomials of the columns of T at the domain H.
Sample random b_1, b_2, b_3, b_4, b_5, b_6
Let

a := (b_1X + b_2)Z_H + a'

b := (b_3X + b_4)Z_H + b'

c := (b_5X + b_6)Z_H + c'

Compute [a]_1, [b]_1, [c]_1 and add them to the transcript.
```

The multiples of $Z_H$ that are added to $a', b', c'$ are the called the blindings. In subsequent rounds the polynomials $a, b, c$ are opened at a point chosen by the verifier. If the blindings are missing, information about the prover's private inputs can be leaked.

In this challenge the participant is given a single proof of the following simple circuit, along with the corresponding values of $b$ and $y$:

```
PRIVATE INPUT:
  x
  h

PUBLIC INPUT:
  b
  y

OUTPUT:
  ASSERT y == h * x + b
```

The flag is `x.representative() || h.representative()`. The objective of the challenge is to utilize the provided information in order to retrieve the private inputs.

## Data provided to participants

Participants get the following values:

1. `y: "3610e39ce7acc430c1fa91efcec93722d77bc4e910ccb195fa4294b64ecb0d35"`,
1. `b: "1b0871ce73e72c599426228e37e7469be9f4fa0b7c9dae950bb77539ca9ebb0f"`.

They also get access to the following files:

1. `src/sith_generate_proof.rs` (this file has flags and toxic waste replaced by `???`)
1. `src/circuit.rs`
1. `srs`
1. `proof`

The files `srs` and `proof` can be deserialized using Lambdaworks methods as follows.

```rust
use std::{fs, io::{BufReader, Read}};
use lambdaworks_plonk::prover::Proof;
use lambdaworks_crypto::commitments::kzg::StructuredReferenceString;
use lambdaworks_math::traits::{Deserializable, Serializable};
use crate::sith_generate_proof::{SithProof, SithSRS};

fn read_challenge_data_from_files() -> (SithSRS, SithProof) {
    // Read proof from file
    let f = fs::File::open("./proof").unwrap();
    let mut reader = BufReader::new(f);
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer).unwrap();
    let proof = Proof::deserialize(&buffer).unwrap();

    // Read SRS from file
    let f = fs::File::open("./srs").unwrap();
    let mut reader = BufReader::new(f);
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer).unwrap();
    let srs = StructuredReferenceString::deserialize(&buffer).unwrap();
    (srs, proof)
}
```

## Solution

The solution for the coordinates is:

1. `x: "2194826651b32ca1055614fc6e2f2de86eab941d2c55bd467268e9"`,
1. `h: "432904cca36659420aac29f8dc5e5bd0dd57283a58ab7a8ce4d1ca"`.

The flag is the concatenation of the two: `FLAG: 2194826651b32ca1055614fc6e2f2de86eab941d2c55bd467268e9432904cca36659420aac29f8dc5e5bd0dd57283a58ab7a8ce4d1ca`

## Solution description

We'll use the notation of the `lambdaworks_plonk_prover` docs.

By checking the code of the challenge the participants can find the following in `circuit.rs`

```rust
/// Witness generator for the circuit `ASSERT y == x * h + b`
pub fn circuit_witness(
    b: &FrElement,
    y: &FrElement,
    h: &FrElement,
    x: &FrElement,
) -> Witness<FrField> {
    let z = x * h;
    let w = &z + b;
    let empty = b.clone();
    Witness {
        a: vec![
            b.clone(),
            y.clone(),
            x.clone(),
            b.clone(),
            w.clone(),
            empty.clone(),
            empty.clone(),
            empty.clone(),
        ],
        ...
```

This code reveals that the way prover constructs the $V$ matrix is

| A   | B   | C   |
| --- | --- | --- |
| b   | -   | -   |
| y   | -   | -   |
| x   | h   | z   |
| b   | z   | w   |
| w   | y   | -   |
| -   | -   | -   |
| -   | -   | -   |
| -   | -   | -   |

Where `-` are empty values. The PLONK implementation of `lambdaworks-plonk` requires the empty values to be filled in with the first public input. So in this case the values `-` will be replaced by $b$. This can be seen directly from the code of the challenge

Therefore, the polynomial $a'$, being the interpolation of the column `A` is

$$a' = b L_1 + y L_2 + x L_3 + b L_4 + w L_5 + b L_6 + b L_7 + b L_8,$$

where $L_i$ is the $i$-th polynomial of the Lagrange basis. Also, the value $w$ is equal to $y$. That can be seen from the code and the fact that the last row of the $V$ matrix corresponds to the assertion of the actual output of the circuit being equal to the claimed output $y$.

During the proof, the verifier sends a challenge $\zeta$ and the prover opens, among other things, the polynomial $a$ at $\zeta$. Since the implementation of the challenge does not include blindings, $a(\zeta) = a'(\zeta)$ and we get

$$a(\zeta) = b L_1(\zeta) + y L_2(\zeta) + x L_3(\zeta) + b L_4(\zeta) + y L_5(\zeta) + b L_6(\zeta) + b L_7(\zeta) + b L_8(\zeta).$$

All the terms in this expression are known to the participants except for $x$, which can be cleared from the equation. To do so the participants need to know how to recover the challenges to get $\zeta$ and how to compute the Lagrange polynomials evaluated at it. The second private input $h$ can be computed as $h = (y - b) / x$.

## Test

A test with the above solution is given in `solution.rs`. To make it pass, lines 25 and 26 of `sith_generate_proof.rs` need to be replaced by the following

```rust
pub const FLAG1: &str = "2194826651b32ca1055614fc6e2f2de86eab941d2c55bd467268e9";
pub const FLAG2: &str = "432904cca36659420aac29f8dc5e5bd0dd57283a58ab7a8ce4d1ca";
```
