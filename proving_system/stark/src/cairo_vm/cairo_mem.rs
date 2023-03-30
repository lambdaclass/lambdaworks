use super::errors::CairoImportError;
use lambdaworks_math::{traits::ByteConversion, unsigned_integer::element::U256};
use std::fs;
#[derive(Clone, Debug, PartialEq)]
pub struct CairoMemoryCell {
    pub address: u64,
    // This could also be a StarkField
    // Since we need to convert everything to StarkFields
    // later, we won't do it here
    pub value: U256,
}
#[derive(Clone, Debug, PartialEq)]
pub struct CairoMemory {
    pub cells: Vec<CairoMemoryCell>,
}

impl CairoMemory {
    fn from_bytes_le(bytes: &[u8]) -> Result<Self, CairoImportError> {
        // Each row is an 8 bytes address
        // and a value of 32 bytes (which is a field)
        const ROW_SIZE: usize = 8 + 32;

        if bytes.len() % ROW_SIZE != 0 {
            return Err(CairoImportError::IncorrectNumberOfBytes);
        }
        let num_rows = bytes.len() / ROW_SIZE;

        let mut cells: Vec<CairoMemoryCell> = Vec::with_capacity(num_rows);

        for i in 0..num_rows {
            cells.push(CairoMemoryCell {
                address: u64::from_le_bytes(
                    bytes[i * ROW_SIZE..i * ROW_SIZE + 8].try_into().unwrap(),
                ),
                value: U256::from_bytes_le(
                    bytes[i * ROW_SIZE + 8..i * ROW_SIZE + 40]
                        .try_into()
                        .unwrap(),
                )
                .unwrap(),
            })
        }

        Ok(Self { cells })
    }

    pub fn from_file(path: &str) -> Result<Self, CairoImportError> {
        let data = fs::read(path)?;
        Self::from_bytes_le(&data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_indexes_are_contiguos_in_bytes_of_mul_program() {
        /*
        Hex from the trace of the following cairo program

        func main() {
            let x = 2;
            let y = 3;
            assert x * y = 6;
            return();
        }

        Generated with:

        cairo-compile multiply.cairo --output multiply.out

        cairo-run --layout all --trace_file trace.out --memory_file mem.out --program multiply.out

        xxd -p mem.out
         */

        let bytes = hex::decode("01000000000000000080ff7f01800648000000000000000000000000000000000000000000000000020000000000000006000000000000000000000000000000000000000000000000000000000000000300000000000000ff7fff7f01800640000000000000000000000000000000000000000000000000040000000000000006000000000000000000000000000000000000000000000000000000000000000500000000000000fe7fff7fff7f8b20000000000000000000000000000000000000000000000000060000000000000009000000000000000000000000000000000000000000000000000000000000000700000000000000090000000000000000000000000000000000000000000000000000000000000008000000000000000600000000000000000000000000000000000000000000000000000000000000").unwrap();

        let memory = CairoMemory::from_bytes_le(&bytes).unwrap();

        for (i, cell) in memory.cells.iter().enumerate() {
            assert_eq!(cell.address, (i + 1) as u64);
        }
    }

    #[test]
    fn test_wrong_amount_of_bytes_gives_err() {
        let bytes = hex::decode("01000000000000000080ff7f01800648000000000000000000000000000000000000000000000000020000000000000006000000000000000000000000000000000000000000000000000000000000000300000000000000ff7fff7f01800640000000000000000000000000000000000000000000000000040000000000000006000000000000000000000000000000000000000000000000000000000000000500000000000000fe7fff7fff7f8b2000000000000000000000000000000000000000000000000006000000000000000900000000000000000000000000000000000000000000000000000000000000070000000000000009000000000000000000000000000000000000000000000000000000000000000800000000000000060000000000000000000000000000000000000000000000000000000000000088").unwrap();

        match CairoMemory::from_bytes_le(&bytes) {
            Err(CairoImportError::IncorrectNumberOfBytes) => (),
            Err(_) => panic!(),
            Ok(_) => panic!(),
        }
    }

    #[test]
    fn mem_indexes_are_contiguos_when_loading_from_file_mul_program() {
        let base_dir = env!("CARGO_MANIFEST_DIR");
        dbg!(base_dir);
        let dir = base_dir.to_owned() + "/src/cairo_vm/test_data/mul_mem.out";

        let memory = CairoMemory::from_file(&dir).unwrap();

        dbg!(memory.clone());

        for (i, cell) in memory.cells.iter().enumerate() {
            assert_eq!(cell.address, (i + 1) as u64);
        }
    }
}
