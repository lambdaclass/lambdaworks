const OFF_DST_OFF: i64 = 0;
const OFF_OP0_OFF: i64 = 16;
const OFF_OP1_OFF: i64 = 32;

// Taken from cairo-rs
// fn decode_offset(offset: i64) -> isize {
//     let vectorized_offset: [u8; 8] = offset.to_le_bytes();
//     let offset_16b_encoded = u16::from_le_bytes([vectorized_offset[0], vectorized_offset[1]]);
//     let complement_const = 0x8000u16;
//     let (offset_16b, _) = offset_16b_encoded.overflowing_sub(complement_const);
//     isize::from(offset_16b as i16)
// }
