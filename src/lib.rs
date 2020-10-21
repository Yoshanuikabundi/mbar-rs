#![warn(rust_2018_idioms, missing_docs, missing_debug_implementations)]
#![allow(incomplete_features)]
#![feature(const_generics)]

//! The multistate Bennett acceptance ratio (MBAR) method for the analysis of equilibrium samples
//! from multiple arbitrary thermodynamic states in computing equilibrium expectations, free energy
//! differences, potentials of mean force, and entropy and enthalpy contributions.
//!
//! Please reference the following if you use this code in your research:
//!
//! [1] Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple
//! equilibrium states. J. Chem. Phys. 129:124105, 2008. <http://dx.doi.org/10.1063/1.2978177>

#[macro_use]
extern crate derive_builder;

use numpy::PyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use std::error::Error;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// `InitMBar::BAR` works best when the states are ordered such that adjacent states maximize
/// the overlap between states. Its up to the user to arrange the states in such an order, or at
/// least close to such an order. If you are uncertain what the order of states should be, or if
/// it does not make sense to think of states as adjacent, then choose `InitMBar::Zeros`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InitialFreeEnergies<const K: usize> {
    /// Use the specified free energy values
    Specified([f64; K]),
    /// Initialize all free energies to zero
    Zeros,
    /// Use BAR between the pairwise state to initialize the free energies.
    ///
    /// Eventually, should specify a path; for now, it just does it zipping up the states.
    BAR,
}

impl<const K: usize> Default for InitialFreeEnergies<K> {
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
/// [1] Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple
/// equilibrium states. J. Chem. Phys. 129:124105, 2008 http://dx.doi.org/10.1063/1.2978177
#[derive(Builder, Debug)]
#[builder(build_fn(validate = "Self::validate", name = "build_inner", private))]
pub struct MBar<const N_TOT: usize, const K: usize> {
    /// `u_kn[k][n]` is the reduced potential energy of configuration n evaluated at state `k`
    u_kn: [[f64; N_TOT]; K],

    /// `n_k[k]` is the number of uncorrelated snapshots sampled from state `k`
    ///
    /// We assume that the states are ordered such that the first `n_k` are from the first state, the
    /// 2nd `n_k` the second state, and so forth. This only becomes important for BAR – MBAR does not
    /// care which samples are from which state. We should eventually allow this assumption to be
    /// overwritten by parameters passed from above, once u_kln is phased out.
    n_k: [usize; K],

    /// Set to limit the maximum number of iterations performed
    #[builder(default = "1000")]
    maximum_iterations: usize,

    /// Set to determine the relative tolerance convergence criteria
    #[builder(default = "1.0e-6")]
    relative_tolerance: f64,

    /// Set to True if verbose debug output is desired
    #[builder(default = "false")]
    verbose: bool,

    /// Set to the initial dimensionless free energies to use as a guess
    #[builder(default)]
    initial_free_energies: InitialFreeEnergies<K>,

    /// Which state is each x from?
    ///
    /// Usually doesn’t matter, but does for BAR. We assume the samples are in K order (the first
    /// `n_k[0]` samples are from the 0th state, the next `n_k[1]` samples from the 1st state, and
    /// so forth.
    #[builder(setter(strip_option), default)]
    x_kindices: Option<Vec<usize>>,

    #[builder(setter(skip), default = "Python::with_gil(|py| py.None())")]
    mbar_obj: PyObject,
}

impl<const N_TOT: usize, const K: usize> MBarBuilder<N_TOT, K> {
    fn validate(&self) -> std::result::Result<(), String> {
        if let Some(n_k) = self.n_k {
            if N_TOT != n_k.iter().sum() {
                return Err("n_k.sum() must equal the total number of samples (N_TOT)".to_string());
            }
        };

        Ok(())
    }

    /// Build and initialise the MBAR implementation
    pub fn build(&self) -> Result<MBar<N_TOT, K>> {
        let mut new = self.build_inner()?;

        let u_kn_refs: Vec<&[f64]> = new.u_kn.iter().map(|inner| inner.as_ref()).collect();

        new.mbar_obj = Python::with_gil(|py| -> PyResult<PyObject> {
            let mbar = py.import("pymbar")?.get("MBAR")?;
            let kwargs: &PyDict = PyDict::new(py);
            kwargs.set_item("u_kn", u_kn_refs)?;
            kwargs.set_item("N_k", new.n_k.as_ref())?;
            kwargs.set_item("maximum_iterations", new.maximum_iterations)?;
            kwargs.set_item("relative_tolerance", new.relative_tolerance)?;
            kwargs.set_item("verbose", new.verbose)?;
            kwargs.set_item("x_kindices", new.x_kindices.clone())?;

            match new.initial_free_energies {
                InitialFreeEnergies::Specified(energies) => {
                    kwargs.set_item("initial_f_k", energies.as_ref())?;
                }
                InitialFreeEnergies::Zeros => {
                    kwargs.set_item("initialize", "zeros")?;
                }
                InitialFreeEnergies::BAR => {
                    kwargs.set_item("initialize", "BAR")?;
                }
            }

            let mbar = PyAny::call(mbar, (), Some(kwargs))?;

            Ok(mbar.to_object(py))
        })?;

        Ok(new)
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

impl<const N_TOT: usize, const K: usize> MBar<N_TOT, K> {
    /// Get a new builder for the `MBar` struct. `MBar` can only be constructed via the builder.
    pub fn builder() -> MBarBuilder<N_TOT, K> {
        MBarBuilder::default()
    }

    /// Compute the free energy of occupying a number of bins.
    ///
    /// This implementation computes the expectation of an indicator-function observable for each bin.
    ///
    /// # Parameters
    ///
    /// `u_n: [f64; N_TOT]` --- `u_n[n]` is the reduced potential energy of snapshot `n` of state `k`
    /// `for which the PMF is to be computed.
    ///
    /// `bin_n: [usize; N_TOT]` --- `bin_n[n]` is the bin index of snapshot `n` of state `k`.
    /// `bin_n` can assume a value in 0..n_bins
    ///
    /// `n_bins: usize` --- The number of bins
    ///
    /// `uncertainties: PmfUncertainties` --- Method for reporting uncertainties
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
        u_n: [f64; N_TOT],
        bin_n: [usize; N_TOT],
        n_bins: usize,
        uncertainties: PmfUncertainties,
    ) -> Result<Pmf> {
        Ok(Python::with_gil(|py| -> PyResult<Pmf> {
            let mbar = &self.mbar_obj;

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

            let u_n_py = PyArray::from_exact_iter(py, u_n.iter().map(|&n| n));
            let bin_n_py = PyArray::from_exact_iter(py, bin_n.iter().map(|&n| n));

            let pmf_result =
                mbar.call_method(py, "computePMF", (u_n_py, bin_n_py, n_bins), Some(kwargs))?;
            let pmf_dict = pmf_result.as_ref(py);

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
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn build_mbar() {
        let mbar = MBarBuilder::<6, 3>::default()
            .u_kn([
                [1.4, 2.3, 3.7, 4.1, 7.7, 9.1],
                [-1.6, -2.3, 9.7, 34.1, 27.7, 19.1],
                [4.4, 7.3, 9.7, 8.1, 4.7, 3.1],
            ])
            .n_k([2, 2, 2])
            .build_inner()
            .unwrap();

        assert_eq!(
            mbar.u_kn,
            [
                [1.4, 2.3, 3.7, 4.1, 7.7, 9.1],
                [-1.6, -2.3, 9.7, 34.1, 27.7, 19.1],
                [4.4, 7.3, 9.7, 8.1, 4.7, 3.1],
            ]
        );
        assert_eq!(mbar.n_k, [2, 2, 2]);
        assert_eq!(mbar.maximum_iterations, 1000);
        assert_eq!(mbar.relative_tolerance, 1.0e-6);
        assert_eq!(mbar.verbose, false);
        assert_eq!(mbar.initial_free_energies, InitialFreeEnergies::Zeros);
        assert_eq!(mbar.x_kindices, None);
    }

    #[test]
    fn init_mbar() {
        let mbar = MBarBuilder::<6, 3>::default()
            .u_kn([
                [1.4, 2.3, 3.7, 4.1, 7.7, 9.1],
                [-1.6, -2.3, 9.7, 34.1, 27.7, 19.1],
                [4.4, 7.3, 9.7, 8.1, 4.7, 3.1],
            ])
            .n_k([2, 2, 2])
            .build()
            .unwrap();

        Python::with_gil(|py| {
            let mbar_class = mbar.mbar_obj.getattr(py, "__class__").unwrap();
            let mbar_class_name = mbar_class.getattr(py, "__name__").unwrap();
            assert_eq!(mbar_class_name.extract::<String>(py).unwrap(), "MBAR");
        });
    }
}
