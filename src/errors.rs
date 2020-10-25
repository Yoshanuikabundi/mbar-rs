use pyo3::prelude::PyErr;
use thiserror::Error;

/// Enum for errors in this crate
#[derive(Error, Debug)]
pub enum MBarError {
    /// Error returned when MBarBuilder.build() was called improperly
    #[error("Could not build MBar: {0}")]
    BuilderError(String),

    /// Error returned when a python exception is not handled
    #[error("Unexpected Python exception was not handled")]
    UnhandledPythonException {
        #[allow(missing_docs)]
        #[from]
        source: PyErr,
    },

    /// Error returned when an array is the wrong length
    #[error("Array of length {0} is incorrect; length should be {1}")]
    ArrayLengthMismatch(usize, usize),
}

impl From<String> for MBarError {
    fn from(s: String) -> Self {
        Self::BuilderError(s)
    }
}

/// Result type for the mbar-rs crate
pub type Result<T> = std::result::Result<T, MBarError>;
