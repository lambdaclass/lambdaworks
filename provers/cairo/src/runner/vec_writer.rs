use cairo_vm::felt::Felt252;
use std::io::{self, Write};

pub struct VecWriter<'a> {
    buf_writer: &'a mut Vec<u8>,
}

impl bincode::enc::write::Writer for VecWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> Result<(), bincode::error::EncodeError> {
        self.buf_writer
            .write_all(bytes)
            .expect("Shouldn't fail in memory vector");

        Ok(())
    }
}

impl<'a> VecWriter<'a> {
    pub fn new(vec: &'a mut Vec<u8>) -> Self {
        Self { buf_writer: vec }
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.buf_writer.flush()
    }

    pub fn write_encoded_trace(
        &mut self,
        relocated_trace: &[cairo_vm::vm::trace::trace_entry::TraceEntry],
    ) {
        for entry in relocated_trace.iter() {
            self.buf_writer
                .extend_from_slice(&((entry.ap as u64).to_le_bytes()));
            self.buf_writer
                .extend_from_slice(&((entry.fp as u64).to_le_bytes()));
            self.buf_writer
                .extend_from_slice(&((entry.pc as u64).to_le_bytes()));
        }
    }

    /// Writes a binary representation of the relocated memory.
    ///
    /// The memory pairs (address, value) are encoded and concatenated:
    /// * address -> 8-byte encoded
    /// * value -> 32-byte encoded
    pub fn write_encoded_memory(&mut self, relocated_memory: &[Option<Felt252>]) {
        for (i, memory_cell) in relocated_memory.iter().enumerate() {
            match memory_cell {
                None => continue,
                Some(unwrapped_memory_cell) => {
                    self.buf_writer.extend_from_slice(&(i as u64).to_le_bytes());
                    self.buf_writer
                        .extend_from_slice(&unwrapped_memory_cell.to_le_bytes());
                }
            }
        }
    }
}
