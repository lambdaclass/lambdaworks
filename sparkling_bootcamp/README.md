# Lambda's Sparkling Water Bootcamp - Repo for challenges and learning path

Public repository for exercises, challenges and all the needs of the Sparkling Water Bootcamp.

## Week 1 - Forging your tools: Finite Fields

This first week will be focused on the development of one of the building blocks of Cryptography: Finite Fields. 

### Recommended material:
- [An introduction to mathematical cryptography](https://books.google.com.ar/books/about/An_Introduction_to_Mathematical_Cryptogr.html?id=BHuTQgAACAAJ&source=kp_book_description&redir_esc=y) - Chapter 1.
- [Finite Fields](https://www.youtube.com/watch?v=MAhmV_omOwA&list=PLFX2cij7c2PynTNWDBzmzaD6ij170ILbQ&index=8)
- [Constructing finite fields](https://www.youtube.com/watch?v=JPiXFn9WA5Y&list=PLFX2cij7c2PynTNWDBzmzaD6ij170ILbQ&index=6)
- [Cyclic groups](https://www.youtube.com/watch?v=UIhhs38IAGM&list=PLFX2cij7c2PynTNWDBzmzaD6ij170ILbQ&index=3)
- [Summary on Montgomery arithmetic](https://eprint.iacr.org/2017/1057.pdf)
- [Mersenne primes](https://eprint.iacr.org/2023/824.pdf)

### Challenges:
- [Implement Montgomery backend for 32 bit fields](https://github.com/lambdaclass/lambdaworks/issues/538).
- [Implement efficient Mersenne prime backend](https://github.com/lambdaclass/lambdaworks/issues/540).
- [Implement efficient backend for pseudo-Mersenne primes](https://github.com/lambdaclass/lambdaworks/issues/393).
- Compare specific field implementations with ordinary Montgomery arithmetic.

### Cryptography content:
- [Serious Cryptography](https://books.google.com.ar/books/about/Serious_Cryptography.html?id=1D-QEAAAQBAJ&source=kp_book_description&redir_esc=y), Chapters 9 & 10.

### Exercises
- Implement naïve version of RSA.
- $7$ is a generator of the multiplicative group of $Z_p^\star$, where $p = 2^{64} - 2^{32} +1$. Find the generators for the $2^{32}$ roots of unity. Find generators for subgroups of order $2^{16} + 1$ and $257$.
- Define in your own words what is a group, a subgroup, a ring and a field.
- What are the applications of the Chinese Remainder Theorem in Cryptography?
- Find all the subgroups of the multiplicative group of $Z_{29}^\star$

## Supplementary Material
- [Polynomial Secret Sharing](https://decentralizedthoughts.github.io/2020-07-17-polynomial-secret-sharing-and-the-lagrange-basis/)
- [Polynomials over a Field](https://decentralizedthoughts.github.io/2020-07-17-the-marvels-of-polynomials-over-a-field/)
- [Fourier Transform](https://www.youtube.com/watch?v=spUNpyF58BY)
- [Fast Fourier Transform](https://www.youtube.com/watch?v=h7apO7q16V0)

## Week 2 - Enter Elliptic Curves

During the second week we'll continue with Finite Fields and begin with Elliptic Curves and dive deeper into Rust

### Recommended material

- [Moonmath Manual](https://leastauthority.com/community-matters/moonmath-manual/) - Chapter 5, until 5.3
- [Programming Bitcoin](https://books.google.fr/books/about/Programming_Bitcoin.html?id=O2aHDwAAQBAJ&source=kp_book_description&redir_esc=y) - Chapters 2 & 3.
- [Introduction to Mathematical Cryptography](https://books.google.com.ar/books/about/An_Introduction_to_Mathematical_Cryptogr.html?id=BHuTQgAACAAJ&source=kp_book_description&redir_esc=y) - Chapter 5 until 5.5
- [Serious Cryptography](https://books.google.com.ar/books/about/Serious_Cryptography.html?id=1D-QEAAAQBAJ&source=kp_book_description&redir_esc=y) - Chapters 11 & 12.
- [Pairings for Beginners](https://static1.squarespace.com/static/5fdbb09f31d71c1227082339/t/5ff394720493bd28278889c6/1609798774687/PairingsForBeginners.pdf) - Chapters 1 & 2

### Exercises

- Define an elliptic curve element type.
- Implement the basic operations: addition and doubling.
- Implement scalar multiplication.
- Check that the point belongs to the correct subgroup.
- The BLS12-381 elliptic curve is given by the equation $y^2 = x^3 + 4$ and defined over $\mathbb{F}_p$ with p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab. The group generator is given by the point p1 = (0x04, 0x0a989badd40d6212b33cffc3f3763e9bc760f988c9926b26da9dd85e928483446346b8ed00e1de5d5ea93e354abe706c) and the cofactor is $h_1 = 0x396c8c005555e1568c00aaab0000aaab$. Find the generator $g$ of the subgroup of order 
r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001.
- Implement a naïve version of the Diffie - Hellman protocol
- Implement point compression and decompression to store elliptic curve points

### Challenges

- Special CTF challenge (will be revealed later)
- [Implement BN254](https://github.com/lambdaclass/lambdaworks/issues/548)
- Implement Secp256k1
- Implement Ed25519

### Rust Workshop

- [Aguante Rust](https://youtu.be/nYpMbjzb1t8?si=HNanyXYWcu1xDjG5)

## Week 3: Polynomials

### Recommended material

- [Polynomials](https://www.youtube.com/watch?v=HiaJa3yhHTU&list=PLFX2cij7c2PynTNWDBzmzaD6ij170ILbQ&index=6)
- [Lagrange interpolation](https://www.youtube.com/watch?v=REnFOKo9gXs&list=PLFX2cij7c2PynTNWDBzmzaD6ij170ILbQ&index=10)
- [Lagrange interpolation and secret sharing](https://www.youtube.com/watch?v=3g4wZnhl4m8&list=PLFX2cij7c2PynTNWDBzmzaD6ij170ILbQ&index=3)
- [Moonmath](https://leastauthority.com/community-matters/moonmath-manual/) - Chapter 3.4
- [Convolution polynomial rings - Introduction to Mathematical Cryptography](https://books.google.com.ar/books/about/An_Introduction_to_Mathematical_Cryptogr.html?id=BHuTQgAACAAJ&source=kp_book_description&redir_esc=y) - Chapter 6.9

### Supplementary

- [Roots of unity and polynomials](https://www.youtube.com/watch?v=3KK5RuAgOpA&list=PLFX2cij7c2PynTNWDBzmzaD6ij170ILbQ&index=2)
- [Fast Fourier Transform](https://www.youtube.com/watch?v=toj_IoCQE-4)
- [FFT walkthrough](https://www.youtube.com/watch?v=Ty0JcR6Dvis)

### Exercises

- Define a polynomial type.
- Implement basic operations, such as addition, multiplication and evaluation.
- Implement Lagrange polynomial interpolation.
- Implement basic version of Shamir's secret sharing.

### Issue

- [Implement Stockham FFT](http://wwwa.pikara.ne.jp/okojisan/otfft-en/stockham1.html)

## Week 4: STARKs

### Recommended material

- [STARKs by Sparkling Water Bootcamp](https://www.youtube.com/watch?v=cDzTm3clrEo)
- [Lambdaworks Docs](https://github.com/lambdaclass/lambdaworks/tree/main/docs/src/starks)
- [Stark 101](https://github.com/starkware-industries/stark101)
- [Constraints](https://blog.lambdaclass.com/periodic-constraints-and-recursion-in-zk-starks/)
- [Stark 101 - rs](https://github.com/lambdaclass/stark101-rs/)
- [Anatomy of a STARK](https://aszepieniec.github.io/stark-anatomy/)
- [BrainSTARK](https://aszepieniec.github.io/stark-brainfuck/)
- [A summary on FRI low degree testing](https://eprint.iacr.org/2022/1216)
- [STARKs by Risc0](https://dev.risczero.com/reference-docs/about-starks)

### Exercises

- Complete STARK-101

## Week 5: Symmetric encryption

### Recommended material

- [One time pad - Dan Boneh](https://www.youtube.com/watch?v=pQkyFJp2eUg&list=PL58C6Q25sEEHXvACYxiav_lC2DqSlC7Og&index=6)
- [Stream ciphers and pseudorandom generators - Dan Boneh](https://www.youtube.com/watch?v=ZSjTMSvp-eI&list=PL58C6Q25sEEHXvACYxiav_lC2DqSlC7Og&index=7)
- [Attacks - Dan Boneh](https://www.youtube.com/watch?v=Qm8fycVt5v8&list=PL58C6Q25sEEHXvACYxiav_lC2DqSlC7Og&index=8)
- [Semantic security - Dan Boneh](https://www.youtube.com/watch?v=6LFyXO58F4A&list=PL58C6Q25sEEHXvACYxiav_lC2DqSlC7Og&index=11)
- [Block ciphers - Dan Boneh](https://www.youtube.com/watch?v=dzoqxqfpZB4&list=PL58C6Q25sEEHXvACYxiav_lC2DqSlC7Og&index=35)
- [Serious Cryptography](https://books.google.com.ar/books/about/Serious_Cryptography.html?id=1D-QEAAAQBAJ&source=kp_book_description&redir_esc=y) - Chapters 3 - 5.

### Supplementary material

- [AES - NIST](https://nvlpubs.nist.gov/nistpubs/fips/nist.fips.197.pdf)

### Exercises

- Implement AES round function

### Side project - Multilinear polynomials

- [Proofs, Args and ZK](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)

### Mandatory task

- Choose a project: STARKs, Sumcheck protocol or Groth16 (or propose a new project)

### Additional resources for each project

- STARKs: see week 4.
- [Groth16](https://eprint.iacr.org/2016/260.pdf)
- [DIZK - Groth 16](https://eprint.iacr.org/2018/691.pdf)
- [Multilinear polynomials and sumcheck protocol](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)

#### Challenges

- Implement a multilinear polynomial type with all the basic operations.

## Week 6: Interactive proofs and SNARKs

- [Moonmath](https://leastauthority.com/community-matters/moonmath-manual/) Chapters 6 - 8.
- [Proofs, Arguments and Zero Knowledge](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf) Chapters 1 - 5.
- [Overview of modern SNARK constructions](https://www.youtube.com/watch?v=bGEXYpt3sj0)
- [Pinocchio protocol overview](https://www.zeroknowledgeblog.com/index.php/zk-snarks)
- [Pinocchio implementation](https://github.com/lambdaclass/pinocchio_lambda_vm)
- [SNARKs and STARKs](https://zkhack.dev/whiteboard/module-four/)

### Additional material on some proof systems

- [EthSTARK](https://github.com/starkware-libs/ethSTARK/tree/master)
- [EthSTARK - paper](https://eprint.iacr.org/2021/582)
- [STARK paper](https://eprint.iacr.org/2018/046.pdf)
- [DEEP FRI](https://eprint.iacr.org/2019/336)
- [Proximity gaps](https://eprint.iacr.org/2020/654)
- [STARKs by Eli Ben-Sasson I](https://www.youtube.com/watch?v=9VuZvdxFZQo)
- [STARKs by Eli Ben-Sasson II](https://www.youtube.com/watch?v=L7tZeO8ihcQ)

## Week 7: Plonk

- [Plonk](https://eprint.iacr.org/2019/953)
- [Custom gates](https://zkhack.dev/whiteboard/module-five/)
- [Plonk by hand](https://research.metastate.dev/plonk-by-hand-part-1/)
- [Plonk docs in Lambdaworks](https://github.com/lambdaclass/lambdaworks/tree/main/docs/src/plonk)

## Week 8: Lookup arguments

- [Plookup](https://eprint.iacr.org/2020/315.pdf)
- [LogUp and GKR](https://eprint.iacr.org/2023/1284.pdf)
- [Neptune - Permutation Argument](https://neptune.cash/learn/tvm-cross-table-args/)
- [Randomized AIR with preprocessing](https://hackmd.io/@aztec-network/plonk-arithmetiization-air)
- [PlonkUp](https://eprint.iacr.org/2022/086.pdf)
- [Lookups by Ingonyama](https://medium.com/@ingonyama/a-brief-history-of-lookup-arguments-a4eeeeca2749)
- [LogUp](https://eprint.iacr.org/2022/1530.pdf)
- [Lookups - Halo2](https://zcash.github.io/halo2/design/proving-system/lookup.html)

## Week 9: Signatures

- [BLS signatures](https://www.ietf.org/archive/id/draft-irtf-cfrg-bls-signature-05.html#name-introduction-2)
- [Real World Cryptography](https://books.google.com.ar/books/about/Real_World_Cryptography.html?id=Qd5CEAAAQBAJ&source=kp_book_description&redir_esc=y) Chapter 7
- [ECDSA](https://www.rfc-editor.org/rfc/rfc6605.txt)
- [RSA Signature](https://www.ietf.org/rfc/rfc8017.html#section-5.2)

## Week 10: Folding schemes

- [Nova by Justin Drake](https://zkhack.dev/whiteboard/module-fourteen/)
- [Nova](https://eprint.iacr.org/2021/370)
- [SuperNova](https://eprint.iacr.org/2022/1758)
- [ProtoStar](https://eprint.iacr.org/2023/620)
- [ProtoGalaxy](https://eprint.iacr.org/2023/1106)

## Projects

- Implement IPA commitment scheme
- Implement Jacobian coordinates for Elliptic Curves
- Benchmark elliptic curve operations
- Add improvements to fixed base scalar multiplication in Elliptic Curves
- Add BN254 elliptic curve
- Implement Pasta curves
- Implement Lookup arguments for Plonk (Plookup)
- Sumcheck protocol
- Benchmark and optimize multilinear polynomial operations
- Import circuits from gnark or circom to use with Groth16 backend

### Links to repos with solutions to the exercises
- [Naïve ECC](https://github.com/saitunc/naive_ecc)
- [Crypto](https://github.com/irfanbozkurt/crypto)
- [Naïve RSA](https://github.com/WiseMrMusa/rsa-naive)
- [Naïve RSA](https://github.com/Elvis339/naive_rsa)
- [Exercises from weeks 1 & 2](https://github.com/ArpitxGit/sparkling_water_bootcamp/tree/main)
- [Programming bitcoin EC](https://github.com/Elvis339/rbtc)
- [Shamir secret sharing](https://github.com/cliraa/shamir_secret_sharing)
- [Several exercises](https://github.com/ArpitxGit/sparkling_water_bootcamp/tree/main)

### Intended Roadmap

- Finite Fields
- Elliptic Curves
- Polynomials
- Extension fields
- Pairings
- Public key encryption
- Symmetric encryption
- Hash functions
- Signatures
- Authenticated encryption
- SNARKs
- STARKs
