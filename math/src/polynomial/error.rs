#[derive(PartialEq, Eq, Debug)]
pub enum MultilinearError {
    InvalidMergeLength,
    IncorrectNumberofEvaluationPoints,
    ChisAndEvalsMismatch,
}
