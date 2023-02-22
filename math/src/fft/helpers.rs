use super::errors::FFTError;

pub fn log2(n: usize) -> Result<u64, FFTError> {
    if !n.is_power_of_two() {
        return Err(FFTError::InvalidOrder(
            "The order of polynomial should a be power of 2".to_string(),
        ));
    }
    Ok(n.trailing_zeros() as u64)
}

pub fn void_ptr<T>(v: &T) -> *const core::ffi::c_void {
    v as *const T as *const core::ffi::c_void
}

pub fn split_u64_into_u32(n: u64) -> (u32, u32) {
    ((n >> 32) as u32, n as u32)
}
