//! The multistate Bennett acceptance ratio (MBAR) method for the analysis of equilibrium samples
//! from multiple arbitrary thermodynamic states in computing equilibrium expectations, free energy
//! differences, potentials of mean force, and entropy and enthalpy contributions.
//!
//! Please reference the following if you use this code in your research:
//!
//! Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple
//! equilibrium states. J. Chem. Phys. 129:124105, 2008. <http://dx.doi.org/10.1063/1.2978177>

#![warn(rust_2018_idioms, missing_docs, missing_debug_implementations)]

#[macro_use]
extern crate derive_builder;

/// Types and methods for computing the Multistate Bennet Acceptance Ratio method
pub mod mbar;

/// A number of test systems with analytically or numerically computable expectations or free
/// energies we use to validate its implementation. These test systems are also convenient to
/// use if you want to easily generate synthetic data to experiment with the capabilities of
/// `pymbar` and `mbar-rs`.
pub mod testsystems;

/// Error types for the crate
pub mod errors;
