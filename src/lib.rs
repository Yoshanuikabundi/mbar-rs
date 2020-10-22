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

use numpy::{PyArray, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
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

type Result<T> = std::result::Result<T, MBarError>;

/// Define the initial guess for free energies
///
/// `InitMBar::BAR` works best when the states are ordered such that adjacent states maximize
/// the overlap between states. Its up to the user to arrange the states in such an order, or at
/// least close to such an order. If you are uncertain what the order of states should be, or if
/// it does not make sense to think of states as adjacent, then choose `InitMBar::Zeros`.
#[derive(Debug, Clone, PartialEq)]
pub enum InitialFreeEnergies {
    /// Use the specified free energy values
    Specified(Vec<f64>),
    /// Initialize all free energies to zero
    Zeros,
    /// Use BAR between the pairwise state to initialize the free energies.
    ///
    /// Eventually, should specify a path; for now, it just does it zipping up the states.
    BAR,
}

impl Default for InitialFreeEnergies {
    fn default() -> Self {
        Self::Zeros
    }
}

/// Multistate Bennett acceptance ratio method (MBAR) for the analysis of multiple equilibrium
/// samples.
///
/// # Notes
///
/// Note that this method assumes the data are uncorrelated.
///
/// Correlated data must be subsampled to extract uncorrelated (effectively independent) samples.
///
/// # References
///
/// 1. Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple
/// equilibrium states. J. Chem. Phys. 129:124105, 2008 <http://dx.doi.org/10.1063/1.2978177>
#[derive(Builder, Debug)]
#[builder(build_fn(validate = "Self::validate", name = "build_inner", private))]
pub struct MBar {
    /// `u_kn[k][n]` is the reduced potential energy of configuration n evaluated at state `k`
    u_kn: ndarray::Array2<f64>,

    /// `n_k[k]` is the number of uncorrelated snapshots sampled from state `k`
    ///
    /// We assume that the states are ordered such that the first `n_k` are from the first state, the
    /// 2nd `n_k` the second state, and so forth. This only becomes important for BAR --- MBAR does
    /// not care which samples are from which state. We should eventually allow this assumption to
    /// be overwritten by parameters passed from above, once u_kln is phased out.
    n_k: ndarray::Array1<usize>,

    /// Set to limit the maximum number of iterations performed
    #[builder(default = "1000")]
    maximum_iterations: usize,

    /// Set to determine the relative tolerance convergence criteria
    #[builder(default = "1.0e-6")]
    relative_tolerance: f64,

    /// Set to the initial dimensionless free energies to use as a guess
    #[builder(default)]
    initial_free_energies: InitialFreeEnergies,

    /// Which state is each x from?
    ///
    /// Usually doesn’t matter, but does for BAR. We assume the samples are in K order (the first
    /// `n_k[0]` samples are from the 0th state, the next `n_k[1]` samples from the 1st state, and
    /// so forth.
    #[builder(setter(strip_option), default)]
    x_kindices: Option<Vec<usize>>,

    /// Set to True if verbose debug output is desired
    #[builder(setter(skip), default = "false")]
    verbose: bool,

    /// Pointer to the MBAR object on Python's heap
    ///
    /// After Self is built, this should always be a valid pointer to an MBAR object. This is
    /// enforced by it being a private field and MBar structs only being constructable via the
    /// builder pattern.
    #[builder(setter(skip), default = "Python::with_gil(|py| py.None())")]
    mbar_obj: PyObject,
}

impl MBarBuilder {
    fn validate(&self) -> std::result::Result<(), String> {
        if let (Some(u_kn), Some(n_k)) = (&self.u_kn, &self.n_k) {
            let k = u_kn.len_of(ndarray::Axis(0));
            let n_tot = u_kn.len_of(ndarray::Axis(1));

            if n_tot != n_k.iter().sum() {
                return Err(format!(
                    "n_k.sum() must equal the total number of samples ({})",
                    n_tot
                ));
            }

            if k != n_k.len() {
                return Err(format!(
                    "n_k's length must equal the number of states ({})",
                    k
                ));
            }
        }

        Ok(())
    }

    /// Build and initialise the MBAR implementation and print progress to STDOUT
    pub fn build_verbose(&self) -> Result<MBar> {
        let mut new = self.build_inner()?;
        new.verbose = true;
        new.init()
    }

    /// Build and initialise the MBAR implementation
    pub fn build(&self) -> Result<MBar> {
        let mut new = self.build_inner()?;
        new.verbose = false;
        new.init()
    }
}

/// Method for reporting uncertainties for PMFs
#[derive(Debug)]
pub enum PmfUncertainties {
    /// The uncertainties in the free energy difference with lowest point on PMF are reported
    FromLowest,
    /// The uncertainties in the free energy difference with the specified state are reported
    FromSpecified(usize),
    /// The normalization $\sum_i p_i = 1$ is used to determine uncertainties spread out through the
    /// PMF
    FromNormalization,
    /// The nbins × nbins matrix $df_ij$ of uncertainties in free energy differences is returned
    /// instead of $df_i$. Doesn't seem to work.
    AllDifferences,
}

impl Default for PmfUncertainties {
    fn default() -> Self {
        Self::FromLowest
    }
}

/// A Potential of Mean Force
#[derive(Debug)]
pub struct Pmf {
    /// `f_i[i]` is the dimensionless free energy of bin `i`, relative to the state of lowest free energy
    pub f_i: Vec<f64>,
    /// `df_i[i]` is the uncertainty in the difference of `f_i` for uncertainty options other than AllDifferences
    pub df_i: Option<Vec<f64>>,
    /// `df_ij[i][j]` is the uncertainty in the difference of `f_i` for AllDifferences
    pub df_ij: Option<Vec<Vec<f64>>>,
}

impl MBar {
    /// Initialise the MBAR in Python; called by build methods
    fn init(mut self) -> Result<Self> {
        Python::with_gil(|py| {
            let mbar = py.import("pymbar")?.get("MBAR")?;
            let kwargs: &PyDict = PyDict::new(py);
            kwargs.set_item("u_kn", PyArray::from_array(py, &self.u_kn))?;
            kwargs.set_item(
                "N_k",
                PyArray::from_exact_iter(py, self.n_k.iter().copied()),
            )?;
            kwargs.set_item("maximum_iterations", self.maximum_iterations)?;
            kwargs.set_item("relative_tolerance", self.relative_tolerance)?;
            kwargs.set_item("verbose", self.verbose)?;
            kwargs.set_item("x_kindices", self.x_kindices.clone())?;

            match &self.initial_free_energies {
                InitialFreeEnergies::Specified(energies) => {
                    if energies.len() != self.k() {
                        return Err(MBarError::ArrayLengthMismatch(energies.len(), self.k()));
                    }
                    let energies_py = PyArray::from_exact_iter(py, energies.iter().copied());
                    kwargs.set_item("initial_f_k", energies_py)?;
                }
                InitialFreeEnergies::Zeros => {
                    kwargs.set_item("initialize", "zeros")?;
                }
                InitialFreeEnergies::BAR => {
                    kwargs.set_item("initialize", "BAR")?;
                }
            }

            self.mbar_obj = PyAny::call(mbar, (), Some(kwargs))?.to_object(py);
            Ok(self)
        })
    }

    /// Get a new builder for the `MBar` struct. `MBar` can only be constructed via the builder.
    pub fn builder() -> MBarBuilder {
        MBarBuilder::default()
    }

    /// $N_{tot}$, the total number of snapshots from all states
    pub fn n_tot(&self) -> usize {
        self.u_kn.len_of(ndarray::Axis(1))
    }

    /// $K$, the total number of thermodynamic states
    pub fn k(&self) -> usize {
        self.u_kn.len_of(ndarray::Axis(0))
    }

    /// Compute the free energy of occupying a number of bins.
    ///
    /// This implementation computes the expectation of an indicator-function observable for each bin.
    ///
    /// # Parameters
    ///
    /// * `u_n[n]` is the reduced potential energy of snapshot `n` of state `k`
    /// for which the PMF is to be computed.
    ///
    /// * `bin_n[n]` is the bin index of snapshot `n` of state `k` and is in `0..n_bins`
    ///
    /// * `n_bins` is the number of bins. No bin may be empty of snapshots
    ///
    /// * `uncertainties` is the method for reporting uncertainties
    ///
    /// # Notes
    ///
    /// - All bins must have some samples in them from at least one of the states – this will not
    ///   work if `bin_n.sum(0) == 0`. Empty bins should be removed before calling `compute_pmf()`.
    /// - This method works by computing the free energy of localizing the system to each bin for
    ///   the given potential by aggregating the log weights for the given potential.
    /// - To estimate uncertainties, the N×K weight matrix $W_nk$ is augmented to be N×(K+`n_bins`)
    ///   in order to accomodate the normalized weights of states where the potential is given by
    ///   `u_kn` within each bin and infinite potential outside the bin. The uncertainties with
    ///   respect to the bin of lowest free energy are then computed in the standard way.
    pub fn compute_pmf(
        &self,
        u_n: &[f64],
        bin_n: &[usize],
        n_bins: usize,
        uncertainties: PmfUncertainties,
    ) -> Result<Pmf> {
        if u_n.len() != self.n_tot() {
            return Err(MBarError::ArrayLengthMismatch(u_n.len(), self.n_tot()));
        }
        if bin_n.len() != self.n_tot() {
            return Err(MBarError::ArrayLengthMismatch(bin_n.len(), self.n_tot()));
        }

        Ok(Python::with_gil(|py| -> PyResult<Pmf> {
            let kwargs: &PyDict = PyDict::new(py);
            kwargs.set_item("return_dict", true)?;
            match uncertainties {
                PmfUncertainties::FromSpecified(ref_state) => {
                    kwargs.set_item("uncertainties", "from-specified")?;
                    kwargs.set_item("pmf_reference", ref_state)?;
                }
                PmfUncertainties::FromLowest => {
                    kwargs.set_item("uncertainties", "from-lowest")?;
                }
                PmfUncertainties::FromNormalization => {
                    kwargs.set_item("uncertainties", "from-normalization")?;
                }
                PmfUncertainties::AllDifferences => {
                    kwargs.set_item("uncertainties", "all-differences")?;
                }
            }

            let u_n_py = PyArray::from_exact_iter(py, u_n.iter().copied());
            let bin_n_py = PyArray::from_exact_iter(py, bin_n.iter().copied());

            let pmf_dict = self.mbar_obj.as_ref(py).call_method(
                "computePMF",
                (u_n_py, bin_n_py, n_bins),
                Some(kwargs),
            )?;

            let f_i = pmf_dict.get_item("f_i")?.extract()?;
            Ok(match uncertainties {
                PmfUncertainties::AllDifferences => {
                    let df_ij = pmf_dict.get_item("df_ij")?.extract()?;
                    Pmf {
                        f_i,
                        df_ij: Some(df_ij),
                        df_i: None,
                    }
                }
                _ => {
                    let df_i = pmf_dict.get_item("df_i")?.extract()?;
                    Pmf {
                        f_i,
                        df_i: Some(df_i),
                        df_ij: None,
                    }
                }
            })
        })?)
    }

    /// Retrieve the weight matrix $W_nk$
    pub fn w_nk(&self) -> ndarray::Array2<f64> {
        Python::with_gil(|py| {
            self.mbar_obj
                .as_ref(py)
                .call_method0("W_nk")
                .expect("Exception during infallible Python method")
                .extract::<&PyArray2<f64>>()
                .expect("Unexpected return type from Python method")
                .to_owned_array()
        })
    }
}
