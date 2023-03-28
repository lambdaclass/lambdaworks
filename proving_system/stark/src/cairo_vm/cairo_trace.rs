use std::fs;
use thiserror::Error;

#[derive(PartialEq, Clone, Debug)]
pub struct CairoTrace {
    rows: Vec<RegistersState>,
}

#[derive(PartialEq, Clone, Debug)]
struct RegistersState {
    pc: u64,
    fp: u64,
    ap: u64,
}

#[derive(Error, Debug)]
pub enum CairoTraceError {
    #[error("Bytes should be a multiple of 24")]
    IncorrectNumberOfBytes,
    #[error("IO Error")]
    FileError(#[from] std::io::Error)
}

impl CairoTrace {

    fn from_bytes_le(bytes: &[u8]) -> Result<Self,CairoTraceError> {

        // Each row of the trace is a RegisterState
        // ap, fp, pc, each 8 bytes long (u64)
        const ROW_SIZE: usize = 8 * 3;

        if bytes.len() % ROW_SIZE != 0 {
            return Err(CairoTraceError::IncorrectNumberOfBytes);
        }
        let num_rows = bytes.len() / ROW_SIZE;

        

        let mut rows: Vec<RegistersState> = Vec::with_capacity(num_rows);

        for i in 0..num_rows {
            rows.push(RegistersState {
                ap: u64::from_le_bytes(bytes[i * ROW_SIZE..i * ROW_SIZE + 8].try_into().unwrap()),
                fp: u64::from_le_bytes(bytes[i * ROW_SIZE + 8..i * ROW_SIZE + 16].try_into().unwrap()),
                pc: u64::from_le_bytes(bytes[i * ROW_SIZE + 16..i * 24 + ROW_SIZE].try_into().unwrap()),
            })
        }

        Ok(Self {
            rows
        })
    }

    fn from_file(path: &str) -> Result<Self, CairoTraceError> {
        let data = fs::read(path)?;
        Ok(Self::from_bytes_le(&data)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul_program_gives_expected_trace() {
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

        xxd -p trace.out
         */

        let bytes = hex::decode("080000000000000008000000000000000100000000000000090000000000000008000000000000000300000000000000090000000000000008000000000000000500000000000000").unwrap();

        let trace = CairoTrace::from_bytes_le(&bytes);

        let expected_state0 = RegistersState {
            ap: 8,
            fp: 8,
            pc: 1,
        };

        let expected_state1 = RegistersState {
            ap: 9,
            fp: 8,
            pc: 3,
        };

        let expected_state2 = RegistersState {
            ap: 9,
            fp: 8,
            pc: 5,
        };

        let expected_trace = CairoTrace {
            rows: [expected_state0, expected_state1, expected_state2].to_vec(),
        };

        assert_eq!(trace.unwrap(), expected_trace)
    }


    #[test]
    fn test_wrong_amount_of_bytes_gives_err() {
        let bytes = hex::decode("080000000000").unwrap();

        match CairoTrace::from_bytes_le(&bytes) {
            Err(CairoTraceError::IncorrectNumberOfBytes) => (),
            Err(_) => panic!(),
            Ok(_) => panic!()
        }
    }

    #[test]
    fn test_loads_mul_trace_correctly() {

        let base_dir = env!("CARGO_MANIFEST_DIR");
        dbg!(base_dir);
        let dir = base_dir.to_owned() + "/src/cairo_vm/test_data/mul_trace.out";
        
        let trace = CairoTrace::from_file(&dir).unwrap();

        let expected_state0 = RegistersState {
            ap: 8,
            fp: 8,
            pc: 1,
        };

        let expected_state1 = RegistersState {
            ap: 9,
            fp: 8,
            pc: 3,
        };

        let expected_state2 = RegistersState {
            ap: 9,
            fp: 8,
            pc: 5,
        };

        let expected_trace = CairoTrace {
            rows: [expected_state0, expected_state1, expected_state2].to_vec(),
        };

        dbg!(&trace);
        assert_eq!(trace,expected_trace);
    }
}
