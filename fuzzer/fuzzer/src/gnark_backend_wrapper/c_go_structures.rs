use crate::gnark_backend_wrapper::GnarkBackendError;
use std::ffi::CString;
use std::os::raw::c_char;

#[derive(Debug)]
#[repr(C)]
pub struct GoString {
    pub ptr: *const c_char,
    length: usize,
}

impl TryFrom<&CString> for GoString {
    type Error = GnarkBackendError;

    fn try_from(value: &CString) -> std::result::Result<Self, Self::Error> {
        let ptr = value.as_ptr();
        let length = value.as_bytes().len();
        Ok(Self { ptr, length })
    }
}

#[repr(C)]
pub struct KeyPair {
    pub proving_key: *const c_char,
    pub verifying_key: *const c_char,
}
