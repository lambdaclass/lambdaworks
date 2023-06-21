#[derive(Debug)]
pub enum FieldError {
    DivisionByZero,
    RootOfUnityError(u64),
}
