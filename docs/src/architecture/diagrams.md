# Architecture Diagrams

This page contains visual diagrams explaining lambdaworks' architecture, data flows, and component relationships.

## High-Level Architecture

The following diagram shows the layered architecture of lambdaworks:

```mermaid
graph TB
    subgraph Applications["Application Layer"]
        APP[Your ZK Application]
    end

    subgraph Provers["Proof Systems Layer"]
        STARK[STARK Prover]
        PLONK[PLONK Prover]
        GROTH16[Groth16 Prover]
        GKR[GKR Protocol]
        SUMCHECK[Sumcheck Protocol]
    end

    subgraph Adapters["Adapter Layer"]
        CIRCOM[Circom Adapter]
        ARKWORKS[Arkworks Adapter]
        WINTERFELL[Winterfell Adapter]
    end

    subgraph Crypto["Cryptographic Layer"]
        MERKLE[Merkle Trees]
        HASH[Hash Functions]
        KZG[KZG Commitments]
        FRI[FRI Protocol]
        FIAT[Fiat-Shamir]
    end

    subgraph Math["Mathematical Layer"]
        FIELDS[Finite Fields]
        CURVES[Elliptic Curves]
        POLY[Polynomials]
        FFT[FFT]
        MSM[MSM]
    end

    subgraph Accel["Acceleration Layer"]
        GPU[GPU Backend]
        ASM[Assembly Optimizations]
    end

    APP --> STARK
    APP --> PLONK
    APP --> GROTH16

    STARK --> FRI
    STARK --> MERKLE
    STARK --> HASH

    PLONK --> KZG
    PLONK --> FIAT

    GROTH16 --> KZG
    GROTH16 --> FIAT

    CIRCOM --> GROTH16
    ARKWORKS --> GROTH16
    WINTERFELL --> STARK

    GKR --> SUMCHECK

    MERKLE --> HASH
    KZG --> CURVES
    FRI --> POLY

    HASH --> FIELDS
    CURVES --> FIELDS
    POLY --> FIELDS
    FFT --> FIELDS
    MSM --> CURVES

    FFT --> GPU
    MSM --> GPU
    FIELDS --> ASM
```

## Crate Dependency Graph

This diagram shows the dependency relationships between lambdaworks crates:

```mermaid
graph LR
    subgraph Provers
        STARK[stark-platinum-prover]
        PLONK[lambdaworks-plonk]
        GROTH16[lambdaworks-groth16]
        SUMCHECK[lambdaworks-sumcheck]
        GKR[lambdaworks-gkr]
    end

    subgraph Adapters
        CIRCOM_ADAPTER[circom-adapter]
        ARK_ADAPTER[arkworks-adapter]
        WINTER_ADAPTER[winterfell-adapter]
    end

    CRYPTO[lambdaworks-crypto]
    MATH[lambdaworks-math]
    GPU[lambdaworks-gpu]

    STARK --> CRYPTO
    STARK --> MATH

    PLONK --> CRYPTO
    PLONK --> MATH

    GROTH16 --> CRYPTO
    GROTH16 --> MATH

    SUMCHECK --> MATH
    GKR --> SUMCHECK

    CIRCOM_ADAPTER --> GROTH16
    ARK_ADAPTER --> GROTH16
    WINTER_ADAPTER --> STARK

    CRYPTO --> MATH
    MATH -.-> GPU
```

## Math Module Structure

Detailed view of the `lambdaworks-math` crate organization:

```mermaid
graph TB
    subgraph Math["lambdaworks-math"]
        subgraph Fields["field/"]
            FE[FieldElement]
            TRAITS[IsField Trait]

            subgraph PrimeFields["Prime Fields"]
                STARK252[Stark252]
                BABYBEAR[BabyBear]
                MERSENNE[Mersenne31]
                GOLDILOCKS[Goldilocks]
            end

            subgraph Extensions["Field Extensions"]
                QUADRATIC[Quadratic]
                CUBIC[Cubic]
                DEGREE12[Degree 12]
            end
        end

        subgraph Curves["elliptic_curve/"]
            CURVE_TRAIT[IsEllipticCurve Trait]
            POINT[Point Types]

            subgraph SW["Short Weierstrass"]
                BLS381[BLS12-381]
                BLS377[BLS12-377]
                BN254[BN254]
                SECP[secp256k1]
                PALLAS[Pallas]
                VESTA[Vesta]
            end

            subgraph Edwards["Twisted Edwards"]
                ED448[Ed448]
                BANDERSNATCH[Bandersnatch]
            end
        end

        subgraph Polynomials["polynomial/"]
            UNIVAR[Univariate]
            MULTILIN[Multilinear]
            SPARSE[Sparse]
        end

        subgraph Transform["fft/"]
            FFT_CPU[CPU FFT]
            FFT_GPU[GPU FFT]
            IFFT[Inverse FFT]
        end

        subgraph MSM_MOD["msm/"]
            PIPPENGER[Pippenger]
            NAIVE[Naive MSM]
        end
    end

    FE --> TRAITS
    PrimeFields --> TRAITS
    Extensions --> PrimeFields

    POINT --> CURVE_TRAIT
    SW --> CURVE_TRAIT
    Edwards --> CURVE_TRAIT

    FFT_CPU --> FE
    MSM_MOD --> POINT
```

## STARK Prover Flow

Sequence diagram showing the STARK proof generation process:

```mermaid
sequenceDiagram
    participant User
    participant Prover
    participant AIR
    participant FRI
    participant Transcript

    User->>Prover: prove(trace, public_inputs, options)

    Prover->>AIR: get_constraints()
    AIR-->>Prover: boundary + transition constraints

    Note over Prover: Interpolate trace polynomials

    Prover->>Prover: compute_composition_poly()
    Note over Prover: Combine constraints with random coefficients

    Prover->>Transcript: append(trace_commitment)
    Transcript-->>Prover: challenge (alpha, beta)

    Prover->>FRI: commit(composition_poly)

    loop FRI Rounds
        FRI->>FRI: fold_polynomial()
        FRI->>Transcript: append(layer_commitment)
        Transcript-->>FRI: random_point
    end

    FRI-->>Prover: fri_proof

    Prover->>Transcript: append(deep_quotient_poly)
    Transcript-->>Prover: query_positions

    Prover->>Prover: generate_queries()

    Prover-->>User: StarkProof
```

## Proof Verification Flow

```mermaid
sequenceDiagram
    participant User
    participant Verifier
    participant FRI_Verifier
    participant Transcript

    User->>Verifier: verify(proof, public_inputs)

    Verifier->>Transcript: rebuild_challenges()
    Note over Transcript: Replay Fiat-Shamir transcript

    Verifier->>Verifier: check_boundary_constraints()
    Verifier->>Verifier: check_transition_constraints()

    Verifier->>FRI_Verifier: verify_fri_proof()

    loop Query Verification
        FRI_Verifier->>FRI_Verifier: verify_merkle_path()
        FRI_Verifier->>FRI_Verifier: check_folding_consistency()
    end

    FRI_Verifier-->>Verifier: verified

    Verifier-->>User: true/false
```

## Type Hierarchy

Class diagram showing the main type relationships:

```mermaid
classDiagram
    class IsField {
        <<trait>>
        +add(a, b) BaseType
        +mul(a, b) BaseType
        +sub(a, b) BaseType
        +neg(a) BaseType
        +inv(a) BaseType
        +zero() BaseType
        +one() BaseType
    }

    class IsPrimeField {
        <<trait>>
        +representative() UnsignedInteger
        +from_hex(s) Self
        +to_bytes_be() Vec~u8~
    }

    class FieldElement~F~ {
        -value: F::BaseType
        +new(value) Self
        +square() Self
        +pow(exp) Self
    }

    class IsEllipticCurve {
        <<trait>>
        +BaseField: IsField
        +generator() PointRepresentation
    }

    class IsShortWeierstrass {
        <<trait>>
        +a() FieldElement
        +b() FieldElement
    }

    class ShortWeierstrassPoint~C~ {
        -x: FieldElement
        -y: FieldElement
        -z: FieldElement
        +to_affine() Self
        +operate_with(other) Self
        +operate_with_self(scalar) Self
    }

    class Polynomial~FE~ {
        -coefficients: Vec~FE~
        +evaluate(x) FE
        +interpolate(xs, ys) Self
        +degree() usize
    }

    class MerkleTree~B~ {
        -root: B::Digest
        -nodes: Vec~B::Digest~
        +build(values) Self
        +get_proof(index) MerkleProof
    }

    IsPrimeField --|> IsField
    FieldElement~F~ --> IsField
    IsShortWeierstrass --|> IsEllipticCurve
    ShortWeierstrassPoint~C~ --> IsEllipticCurve
    Polynomial~FE~ --> FieldElement~F~
```

## Proving System Comparison

Comparison of the three main proving systems:

```mermaid
graph TB
    subgraph STARK_System["STARK"]
        STARK_AIR[AIR Definition]
        STARK_TRACE[Execution Trace]
        STARK_FRI[FRI Commitment]
        STARK_HASH[Hash-based]

        STARK_AIR --> STARK_TRACE
        STARK_TRACE --> STARK_FRI
        STARK_FRI --> STARK_HASH
    end

    subgraph PLONK_System["PLONK"]
        PLONK_CIRCUIT[Circuit Definition]
        PLONK_GATES[Gate Constraints]
        PLONK_KZG[KZG Commitment]
        PLONK_PAIRING[Pairing-based]

        PLONK_CIRCUIT --> PLONK_GATES
        PLONK_GATES --> PLONK_KZG
        PLONK_KZG --> PLONK_PAIRING
    end

    subgraph Groth16_System["Groth16"]
        G16_R1CS[R1CS Circuit]
        G16_QAP[QAP Transform]
        G16_TRUSTED[Trusted Setup]
        G16_PAIRING[Pairing-based]

        G16_R1CS --> G16_QAP
        G16_QAP --> G16_TRUSTED
        G16_TRUSTED --> G16_PAIRING
    end
```

## Data Flow: Field Element Operations

```mermaid
flowchart LR
    subgraph Input
        A[User Value]
        B[Hex String]
        C[Bytes]
    end

    subgraph Conversion
        D[to Montgomery Form]
    end

    subgraph Operations
        E[Add]
        F[Multiply]
        G[Invert]
        H[Square]
    end

    subgraph Output
        I[Representative]
        J[to_hex]
        K[to_bytes]
    end

    A --> D
    B --> D
    C --> D

    D --> E
    D --> F
    D --> G
    D --> H

    E --> I
    F --> I
    G --> I
    H --> I

    I --> J
    I --> K
```

## Merkle Tree Structure

```mermaid
graph TB
    ROOT[Root Hash]

    H01[Hash 0-1]
    H23[Hash 2-3]

    H0[Hash 0]
    H1[Hash 1]
    H2[Hash 2]
    H3[Hash 3]

    L0[Leaf 0]
    L1[Leaf 1]
    L2[Leaf 2]
    L3[Leaf 3]

    ROOT --> H01
    ROOT --> H23

    H01 --> H0
    H01 --> H1

    H23 --> H2
    H23 --> H3

    H0 --> L0
    H1 --> L1
    H2 --> L2
    H3 --> L3

    style ROOT fill:#f9f,stroke:#333,stroke-width:2px
    style L0 fill:#bbf,stroke:#333
    style L1 fill:#bbf,stroke:#333
    style L2 fill:#bbf,stroke:#333
    style L3 fill:#bbf,stroke:#333
```

## KZG Commitment Scheme Flow

```mermaid
sequenceDiagram
    participant Setup
    participant Committer
    participant Prover
    participant Verifier

    Note over Setup: Generate SRS (tau powers)
    Setup->>Committer: SRS = {g, tau*g, tau^2*g, ...}

    Committer->>Committer: Compute C = p(tau)*g via MSM
    Committer->>Prover: Commitment C

    Note over Prover: Open at point z
    Prover->>Prover: Compute quotient q(x) = (p(x)-y)/(x-z)
    Prover->>Prover: Compute pi = q(tau)*g
    Prover->>Verifier: (z, y, pi)

    Note over Verifier: Verify using pairing
    Verifier->>Verifier: Check e(C - y*g, g) = e(pi, tau*g - z*g)
    Verifier->>Prover: Accept/Reject
```

## Using These Diagrams

These diagrams are written in Mermaid syntax and can be rendered in:

1. **GitHub**: Automatically renders Mermaid in Markdown files.
2. **mdBook**: Use the `mdbook-mermaid` preprocessor.
3. **VS Code**: Install the Mermaid extension for preview.
4. **Online**: Use [mermaid.live](https://mermaid.live) for editing and export.

To add Mermaid support to your mdBook, add to `book.toml`:

```toml
[preprocessor.mermaid]
command = "mdbook-mermaid"

[output.html]
additional-js = ["mermaid.min.js", "mermaid-init.js"]
```
