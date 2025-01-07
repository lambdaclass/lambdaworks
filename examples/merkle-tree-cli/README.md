<div align="center">

# Lambdaworks Merklee Tree CLI

Simple Merkle Tree CLI that uses [Poseidon hash](https://www.poseidon-hash.info/).

</div>

### Usage:

### To create a Merkle Tree and save it to a file you can use:

```bash
cargo run --release generate-tree <TREE_PATH>
```

The format of a tree `csv` file is as follows:
```
elem1;elem2;...;elemN
```
For example, the provided **`sample_tree.csv`** looks like this:
```
0x12345;0x6789A;0xBCDEF
```

**`generate-tree` example:**

```bash
cargo run --release generate-tree sample_tree.csv
```
This will:
- Generate a `json` file with the tree structure and save it to the same directory as the `csv` file.

- Save the root of the tree in a `txt` file named `<CSV_FILENAME>_root.txt`.

    For example:

    ```
    sample_tree_root.txt
    ```
    will contain 
    ```
    0xa3bbbb9eac9f79d18862b802ea79f87e75efc37d4f4af4464976784c14a851b69c09aa04b1e8a8d1eb9825b713dc6ca
    ```


- Print the root of the tree in the terminal


### To generate proof for a Merkle Tree you can use: 

```bash
cargo run --release generate-proof <TREE_PATH> <POSITION>
```
This will:
- Generate a  `json` file with the proof for the leaf at the specified position and save it to the same directory as the `csv` file.

- Save the value of the leaf in a `txt` file named `<CSV_FILENAME>_leaf_<POSITION>.txt`.


**`generate-proof` example:**

```bash
cargo run --release generate-proof sample_tree.csv 0
```
This will generate:

- `sample_tree_proof_0.json` will contain the proof for the leaf at position 0.

- `sample_tree_leaf_0.txt` will contain the value of the leaf at position 0. For example:
    ```
    0x12345
    ```

### To verify a proof you can use:

```bash
cargo run --release verify-proof <ROOT_PATH> <INDEX> <PROOF_PATH> <LEAF_PATH>
```


**`verify-proof` example:**

```bash
cargo run --release verify-proof sample_tree_root.txt 0 sample_tree_proof_0.json sample_tree_leaf_0.txt
```
