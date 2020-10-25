use crate::errors::*;
use ndarray::{Array1, Array2, Axis};
use numpy::{PyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

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

impl From<&[f64]> for InitialFreeEnergies {
    fn from(slice: &[f64]) -> Self {
        Self::Specified(Vec::from(slice))
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
    u_kn: Array2<f64>,

    /// `n_k[k]` is the number of uncorrelated snapshots sampled from state `k`
    ///
    /// We assume that the states are ordered such that the first `n_k` are from the first state, the
    /// 2nd `n_k` the second state, and so forth. This only becomes important for BAR --- MBAR does
    /// not care which samples are from which state. We should eventually allow this assumption to
    /// be overwritten by parameters passed from above, once u_kln is phased out.
    n_k: Array1<usize>,

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
    x_kindices: Option<Array1<usize>>,

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
            let k = u_kn.len_of(Axis(0));
            let n_tot = u_kn.len_of(Axis(1));

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
            if let Some(x_kindices) = &self.x_kindices {
                kwargs.set_item("x_kindices", PyArray::from_array(py, x_kindices))?;
            }

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
        self.u_kn.len_of(Axis(1))
    }

    /// $K$, the total number of thermodynamic states
    pub fn k(&self) -> usize {
        self.u_kn.len_of(Axis(0))
    }

    /// `u_kn[k][n]` is the reduced potential energy of configuration n evaluated at state `k`
    pub fn u_kn(&self) -> &Array2<f64> {
        &self.u_kn
    }

    /// `n_k[k]` is the number of uncorrelated snapshots sampled from state `k`
    ///
    /// We assume that the states are ordered such that the first `n_k` are from the first state, the
    /// 2nd `n_k` the second state, and so forth. This only becomes important for BAR --- MBAR does
    /// not care which samples are from which state. We should eventually allow this assumption to
    /// be overwritten by parameters passed from above, once u_kln is phased out.
    pub fn n_k(&self) -> &Array1<usize> {
        &self.n_k
    }

    /// Which state is each x from?
    ///
    /// Usually doesn’t matter, but does for BAR. We assume the samples are in K order (the first
    /// `n_k[0]` samples are from the 0th state, the next `n_k[1]` samples from the 1st state, and
    /// so forth.
    pub fn x_kindices(&self) -> &Option<Array1<usize>> {
        &self.x_kindices
    }

    /// Retrieve a copy of the relative dimensionless free energy $f_k$ of states $k$
    pub fn f_k(&self) -> Array1<f64> {
        Python::with_gil(|py| {
            self.mbar_obj
                .as_ref(py)
                .getattr("f_k")
                .expect("Exception retrieving Python attribute MBAR.f_k")
                .extract::<&PyArray1<f64>>()
                .expect("Unexpected type from Python attribute MBAR.f_k")
                .to_owned_array()
        })
    }

    /// Retrieve a copy of the log weight matrix $\ln(W_nk)$
    pub fn log_w_nk(&self) -> Array2<f64> {
        Python::with_gil(|py| {
            self.mbar_obj
                .as_ref(py)
                .getattr("Log_W_nk")
                .expect("Exception retrieving Python attribute MBAR.Log_W_nk")
                .extract::<&PyArray2<f64>>()
                .expect("Unexpected type from Python attribute MBAR.Log_W_nk")
                .to_owned_array()
        })
    }

    /// Retrieve a copy of the weight matrix $W_nk$
    pub fn w_nk(&self) -> Array2<f64> {
        Python::with_gil(|py| {
            self.mbar_obj
                .as_ref(py)
                .call_method0("W_nk")
                .expect("Exception during infallible Python method MBAR.w_nk()")
                .extract::<&PyArray2<f64>>()
                .expect("Unexpected return type from Python method MBAR.w_nk()")
                .to_owned_array()
        })
    }

    /// Compute the effective sample number of each state
    ///
    /// The effective sample number $n_\mathrm{eff}(k)$ is an estimate of how many samples are
    /// contributing to the average at a given state.
    ///
    /// # Returns
    ///
    /// `n_eff[k]` is the estimated number of samples contributing to estimates at each
    /// state k. An estimate to how many samples collected just at state k would result
    /// in similar statistical efficiency as the MBAR simulation. Valid for both sampled
    /// states, in which the weight will be greater than `N_k[k]`, and unsampled states.
    ///
    /// # Notes
    ///
    /// Using Kish (1965) formula (Kish, Leslie (1965). Survey Sampling. New York: Wiley)
    ///
    /// As the weights become more concentrated in fewer observations, the effective sample size
    /// shrinks (<http://healthcare-economist.com/2013/08/22/effective-sample-size/>):
    ///
    /// $$
    ///     n_\mathrm{eff}(k)
    ///         =  \frac{(\sum_{n=1}^N w_k n)^2}{\sum_{n=1}^N w_k n^2}
    ///         =  \frac{1}{\sum_{n=1}^N w_k n^2}
    /// $$
    ///
    /// the effective sample number is most useful to diagnose when there are only a few samples
    /// contributing to the averages.
    pub fn n_eff(&self) -> Array1<f64> {
        self.compute_effective_sample_number(false)
    }

    /// Compute the effective sample number of each state;
    ///
    /// The effective sample number $n_\mathrm{eff}(k)$ is an estimate of how many samples are
    /// contributing to the average at a given state. It also counts the efficiency of the
    /// sampling, which is simply the ratio of the effective number of samples at a given
    /// state to the total number of samples collected.  This is printed in verbose output,
    /// but is not returned for now.
    ///
    /// # Returns
    ///
    /// `n_eff[k]` is the estimated number of samples contributing to estimates at each
    /// state k. An estimate to how many samples collected just at state k would result
    /// in similar statistical efficiency as the MBAR simulation. Valid for both sampled
    /// states, in which the weight will be greater than `N_k[k]`, and unsampled states.
    ///
    /// # Notes
    ///
    /// Using Kish (1965) formula (Kish, Leslie (1965). Survey Sampling. New York: Wiley)
    ///
    /// As the weights become more concentrated in fewer observations, the effective sample size
    /// shrinks (<http://healthcare-economist.com/2013/08/22/effective-sample-size/>):
    ///
    /// $$
    ///     n_\mathrm{eff}(k)
    ///         =  \frac{(\sum_{n=1}^N w_k n)^2}{\sum_{n=1}^N w_k n^2}
    ///         =  \frac{1}{\sum_{n=1}^N w_k n^2}
    /// $$
    ///
    /// the effective sample number is most useful to diagnose when there are only a few samples
    /// contributing to the averages.
    pub fn n_eff_verbose(&self) -> Array1<f64> {
        self.compute_effective_sample_number(true)
    }

    fn compute_effective_sample_number(&self, verbose: bool) -> Array1<f64> {
        Python::with_gil(|py| {
            self.mbar_obj
                .as_ref(py)
                .call_method1("computeEffectiveSampleNumber", (verbose,))
                .expect(
                    "Exception during infallible Python method MBAR.computeEffectiveSampleNumber()",
                )
                .extract::<&PyArray1<f64>>()
                .expect(
                    "Unexpected return type from Python method MBAR.computeEffectiveSampleNumber()",
                )
                .to_owned_array()
        })
    }

    /// Calculate the variance of a weighted sum of free energy differences.
    ///
    /// For example,  $\mathrm{Var}(\sum_i a_i df_i)$
    ///
    /// # Parameters
    ///
    /// `d_ij` : a matrix of standard deviations of the quantities f_i - f_j
    /// `k` : The number of states in each 'chunk', has to be constant
    ///
    /// # Returns
    ///
    /// KxK variance matrix for the sums or differences $\sum_i a_i df_i$
    ///
    /// # Notes
    ///
    /// This derivation is taken from the pymbar documentation and I'm not confident my
    /// interpretation is correct.
    ///
    /// We explicitly lay out the calculations for four variables (where each variable
    /// is a logarithm of a partition function), then generalize.
    /// The uncertainty in the sum of two weighted differences is
    ///
    /// $$
    /// \begin{aligned}
    ///     \mathrm{Var}(a_1(f_{i1} - f_{j1}) + a_2(f_{i2} - f_{j2})) =&\ a_1^2 \mathrm{Var}(f_{i1} - f_{j1}) \\\\
    ///         & + a_2^2 \mathrm{Var}(f_{i2} - f_{j2}) \\\\
    ///         & + 2 a_1 a_2 \mathrm{cov}(f_{i1} - f_{j1}, f_{i2} - f_{j2})
    /// \end{aligned}
    /// $$
    /// $$
    /// \begin{aligned}
    ///     \mathrm{cov}(f_{i1} - f_{j1}, f_{i2} - f_{j2}) =&\ \mathrm{cov}(f_{i1},f_{i2}) \\\\
    ///         & - \mathrm{cov}(f_{i1},f_{j2}) \\\\
    ///         & - \mathrm{cov}(f_{j1},f_{i2}) \\\\
    ///         & + \mathrm{cov}(f_{j1},f_{j2})
    /// \end{aligned}
    /// $$
    ///
    /// call:
    ///
    /// $$
    /// \begin{aligned}
    ///     f_{i1} &= a \\\\
    ///     f_{j1} &= b \\\\
    ///     f_{i2} &= c \\\\
    ///     f_{j2} &= d \\\\
    ///     a_1^2 \mathrm{Var}(a-b) + a_2^2 \mathrm{Var}(c-d) + 2 a_1 a_2 &= \mathrm{cov}(a-b,c-d)
    /// \end{aligned}
    /// $$
    ///
    /// We want
    ///
    /// $$2 \mathrm{cov}(a-b,c-d) = 2 \mathrm{cov}(a,c)-2 \mathrm{cov}(a,d)-2 \mathrm{cov}(b,c)+2 \mathrm{cov}(b,d)$$
    ///
    /// Since
    ///
    /// $$\mathrm{Var}(x-y) = \mathrm{Var}(x) + \mathrm{Var}(y) - 2 \mathrm{cov}(x,y)$$
    ///
    /// It follows
    ///
    /// $$2 \mathrm{cov}(x,y) = -\mathrm{Var}(x-y) + \mathrm{Var}(x) + \mathrm{Var}(y)$$
    ///
    /// So, we get
    ///
    /// $$
    /// \begin{aligned}
    ///     2 \mathrm{cov}(a,c) &= -\mathrm{Var}(a-c) + \mathrm{Var}(a) + \mathrm{Var}(c) \\\\
    ///     -2 \mathrm{cov}(a,d) &= +\mathrm{Var}(a-d) - \mathrm{Var}(a) - \mathrm{Var}(d) \\\\
    ///     -2 \mathrm{cov}(b,c) &= +\mathrm{Var}(b-c) - \mathrm{Var}(b) - \mathrm{Var}(c) \\\\
    ///     2 \mathrm{cov}(b,d) &= -\mathrm{Var}(b-d) + \mathrm{Var}(b) + \mathrm{Var}(d)
    /// \end{aligned}
    /// $$
    ///
    /// adding up, finally :
    ///
    /// $$
    /// \begin{aligned}
    ///     2 \mathrm{cov}(a-b,c-d) =&\ 2 \mathrm{cov}(a,c) \\\\
    ///                              &\ -2 \mathrm{cov}(a,d) \\\\
    ///                              &\ -2 \mathrm{cov}(b,c) \\\\
    ///                              &\ +2 \mathrm{cov}(b,d) \\\\
    ///                             =&\ - \mathrm{Var}(a-c) \\\\
    ///                              &\ + \mathrm{Var}(a-d) \\\\
    ///                              &\ + \mathrm{Var}(b-c) \\\\
    ///                              &\ - \mathrm{Var}(b-d)
    /// \end{aligned}
    /// $$
    ///
    /// $$
    /// \begin{aligned}
    ///     a_1^2 \mathrm{Var}(a-b)+a_2^2 \mathrm{Var}(c-d) + 2 a_1 a_2 \mathrm{cov}(a-b,c-d) =
    ///         &\ a_1^2 \mathrm{Var}(a-b) \\\\
    ///         &\ + a_2^2 \mathrm{Var}(c-d) \\\\
    ///         &\ + a_1a_2 [-\mathrm{Var}(a-c)+\mathrm{Var}(a-d)+\mathrm{Var}(b-c)-\mathrm{Var}(b-d)]
    /// \end{aligned}
    /// $$
    ///
    /// $$
    /// \begin{aligned}
    ///     \mathrm{Var}(a_1(f_{i1} - f_{j1}) + a_2(f_{i2} - f_{j2})) =&\ a_1^2 \mathrm{Var}(f_{i1} - f_{j1}) \\\\
    ///         &\ + a_2^2 \mathrm{Var}(f_{i2} - f_{j2}) \\\\
    ///         &\ + 2a_1 a_2 \mathrm{cov}(f_{i1} - f_{j1}, f_{i2} - f_{j2}) \\\\
    ///         =&\ a_1^2 \mathrm{Var}(f_{i1} - f_{j1}) \\\\
    ///         &\ + a_2^2 \mathrm{Var}(f_{i2} - f_{j2}) \\\\
    ///         &\ + a_1 a_2 [-\mathrm{Var}(f_{i1} - f_{i2}) + \mathrm{Var}(f_{i1} - f_{j2}) + \mathrm{Var}(f_{j1}-f_{i2}) - \mathrm{Var}(f_{j1} - f_{j2})]
    /// \end{aligned}
    /// $$
    ///
    /// assume two arrays of free energy differences, and an array of constant vectors $a$.
    /// we want the variance $\mathrm{Var}(\sum_k a_k (f_{i,k} - f_{j,k}))$
    /// Each set is separated from the other by an offset $K$. The same process applies with
    /// the sum, with the single $\mathrm{Var}$ terms and the pair terms
    pub fn covariance_of_sums(
        &self,
        d_ij: Array2<f64>,
        k: usize,
        a: Array1<f64>,
    ) -> Result<Array2<f64>> {
        Ok(Python::with_gil(|py| -> PyResult<_> {
            Ok(self
                .mbar_obj
                .as_ref(py)
                .call_method1(
                    "computeCovarianceOfSums",
                    (
                        PyArray::from_array(py, &d_ij),
                        k,
                        PyArray::from_array(py, &a),
                    ),
                )?
                .extract::<&PyArray2<f64>>()?
                .to_owned_array())
        })?)
    }

    /// Compute the expectation of an observable of a phase space function.
    ///
    /// Compute the expectation of an observable of a single phase space
    /// function A(x) at all states where potentials are generated.
    ///
    /// # Parameters
    ///
    /// A_n : np.ndarray, float
    ///     A_n (N_max np float64 array) - A_n[n] = A(x_n)
    ///
    /// References
    /// ----------
    /// See Section IV of [1].
    pub fn expectations(
        &self,
        a_n: Array1<f64>,
        params: ExpectationsParams,
    ) -> Result<Expectations> {
        Python::with_gil(|py| {
            let pydict = self.expectations_inner(a_n, params, false)?;
            let pydict = pydict.as_ref(py);
            let theta = pydict
                .get_item("Theta")?
                .extract::<&PyArray2<f64>>()?
                .to_owned_array();
            let mu = pydict
                .get_item("mu")?
                .extract::<&PyArray1<f64>>()?
                .to_owned_array();
            let sigma = pydict
                .get_item("sigma")?
                .extract::<&PyArray1<f64>>()?
                .to_owned_array();
            Ok(Expectations { theta, mu, sigma })
        })
    }

    /// Compute the differences (in expectation) of an observable of a phase space function between states.
    ///
    /// Compute the expectation of an observable of a single phase space
    /// function A(x) at all states where potentials are generated.
    ///
    /// # Parameters
    ///
    /// A_n : np.ndarray, float
    ///     A_n (N_max np float64 array) - A_n[n] = A(x_n)
    ///
    /// References
    /// ----------
    /// See Section IV of [1].
    pub fn differences_in_expectation(
        &self,
        a_n: Array1<f64>,
        params: ExpectationsParams,
    ) -> Result<DifferencesInExpectation> {
        Python::with_gil(|py| {
            let pydict = self.expectations_inner(a_n, params, true)?;
            let pydict = pydict.as_ref(py);
            let theta = pydict
                .get_item("Theta")?
                .extract::<&PyArray2<f64>>()?
                .to_owned_array();
            let mu = pydict
                .get_item("mu")?
                .extract::<&PyArray2<f64>>()?
                .to_owned_array();
            let sigma = pydict
                .get_item("sigma")?
                .extract::<&PyArray2<f64>>()?
                .to_owned_array();
            Ok(DifferencesInExpectation { theta, mu, sigma })
        })
    }

    fn expectations_inner(
        &self,
        a_n: Array1<f64>,
        params: ExpectationsParams,
        differences: bool,
    ) -> PyResult<PyObject> {
        let ExpectationsParams {
            u_kn,
            state_dependent,
            uncertainty_method,
            warning_cutoff,
        } = params;

        Python::with_gil(|py| {
            let kwargs: &PyDict = PyDict::new(py);
            if let Some(u_kn) = u_kn {
                kwargs.set_item("u_kn", PyArray::from_array(py, &u_kn))?;
            };
            kwargs.set_item(
                "output",
                if differences {
                    "differences"
                } else {
                    "averages"
                },
            )?;
            kwargs.set_item("state_dependent", state_dependent)?;

            match uncertainty_method {
                ExpectationsUncertaintyMethod::Approximate => {
                    kwargs.set_item("compute_uncertainty", true)?;
                    kwargs.set_item("uncertainty_method", "approximate")?;
                }
                ExpectationsUncertaintyMethod::Svd => {
                    kwargs.set_item("compute_uncertainty", true)?;
                    kwargs.set_item("uncertainty_method", "svd")?;
                }
                ExpectationsUncertaintyMethod::SvdEw => {
                    kwargs.set_item("compute_uncertainty", true)?;
                    kwargs.set_item("uncertainty_method", "svd-ew")?;
                }
            }

            kwargs.set_item("warning_cutoff", warning_cutoff)?;
            kwargs.set_item("return_theta", true)?;
            kwargs.set_item("return_dict", true)?;

            let expectations_dict = self.mbar_obj.as_ref(py).call_method(
                "computeExpectations",
                (PyArray::from_array(py, &a_n),),
                Some(kwargs),
            )?;

            Ok(expectations_dict.to_object(py))
        })
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

    /// Decompose free energy differences into enthalpy and entropy differences.
    ///
    /// Compute the decomposition of the free energy difference between
    /// states 1 and N into reduced free energy differences, reduced potential
    /// (enthalpy) differences, and reduced entropy (S/k) differences.
    pub fn compute_entropy_and_enthalpy(
        &self,
        params: ExpectationsParams,
    ) -> Result<EntropyEnthalpy> {
        let ExpectationsParams {
            u_kn,
            uncertainty_method,
            warning_cutoff,
            ..
        } = params;

        Python::with_gil(|py| {
            let kwargs: &PyDict = PyDict::new(py);
            if let Some(u_kn) = u_kn {
                kwargs.set_item("u_kn", PyArray::from_array(py, &u_kn))?;
            };

            match uncertainty_method {
                ExpectationsUncertaintyMethod::Approximate => {
                    kwargs.set_item("uncertainty_method", "approximate")?;
                }
                ExpectationsUncertaintyMethod::Svd => {
                    kwargs.set_item("uncertainty_method", "svd")?;
                }
                ExpectationsUncertaintyMethod::SvdEw => {
                    kwargs.set_item("uncertainty_method", "svd-ew")?;
                }
            }

            kwargs.set_item("warning_cutoff", warning_cutoff)?;
            kwargs.set_item("return_theta", true)?;
            kwargs.set_item("return_dict", true)?;

            let pydict = self.mbar_obj.as_ref(py).call_method(
                "computeEntropyAndEnthalpy",
                (),
                Some(kwargs),
            )?;

            let delta_f = pydict
                .get_item("Delta_f")?
                .extract::<&PyArray2<f64>>()?
                .to_owned_array();
            let d_delta_f = pydict
                .get_item("dDelta_f")?
                .extract::<&PyArray2<f64>>()?
                .to_owned_array();
            let delta_u = pydict
                .get_item("Delta_u")?
                .extract::<&PyArray2<f64>>()?
                .to_owned_array();
            let d_delta_u = pydict
                .get_item("dDelta_u")?
                .extract::<&PyArray2<f64>>()?
                .to_owned_array();
            let delta_s = pydict
                .get_item("Delta_s")?
                .extract::<&PyArray2<f64>>()?
                .to_owned_array();
            let d_delta_s = pydict
                .get_item("dDelta_s")?
                .extract::<&PyArray2<f64>>()?
                .to_owned_array();

            Ok(EntropyEnthalpy {
                delta_f,
                d_delta_f,
                delta_u,
                d_delta_u,
                delta_s,
                d_delta_s,
            })
        })
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

/// Method to use to compute uncertainty for expectations
///
/// Method used to compute the asymptotic covariance matrix.
#[derive(Debug)]
pub enum ExpectationsUncertaintyMethod {
    /// Use fast approximate expression from Kong et al.
    ///
    /// This underestimates the true covariance, but may be a good approximation in some cases and
    /// requires no matrix inversions
    Approximate,
    /// Use singular value decomposition based approach to efficiently compute uncertainty
    ///
    /// See Appendix D.1, Eq. D4 in [1].
    Svd,
    /// Use eigenvalue decomposition of W'W in a SVD-based approach to efficiently compute uncertainty
    ///
    /// The eigenvalue decomposition of W'W is used to forego computing the SVD.
    /// See Appendix D.1, Eqs. D4 and D5 of [1].
    SvdEw,
}

/// Parameters for computing expectation values
#[derive(Debug)]
pub struct ExpectationsParams {
    /// `u_kn[k, n]` is the reduced potential of configuration n at state k
    ///
    /// If None (the default) use MBar.u_kn
    pub u_kn: Option<Array2<f64>>,
    /// Are the expectations state dependent? Default is false. Ignored for compute_entropy_and_enthalpy.
    pub state_dependent: bool,
    /// How to compute uncertainties. Default is SvdEw
    pub uncertainty_method: ExpectationsUncertaintyMethod,
    /// Warn if squared-uncertainty is negative and larger in magnitude than this number (default: 1.0e-10)
    pub warning_cutoff: f64,
}

impl Default for ExpectationsParams {
    fn default() -> Self {
        Self {
            u_kn: None,
            state_dependent: false,
            uncertainty_method: ExpectationsUncertaintyMethod::SvdEw,
            warning_cutoff: 1.0e-10,
        }
    }
}

/// Expectations for a value
#[derive(Debug)]
pub struct Expectations {
    /// Covariance matrix of log weights
    pub theta: Array2<f64>,
    /// `mu[i]` is the estimate for the expectation of $A(x)$ for thermodynamic state `i`.
    pub mu: Array1<f64>,
    /// `sigma[i]` is the uncertainty estimate at one standard deviation for mu[i]
    pub sigma: Array1<f64>,
}

/// Expectations for a value expressed as differences between states
#[derive(Debug)]
pub struct DifferencesInExpectation {
    /// Covariance matrix of log weights
    pub theta: Array2<f64>,
    /// `mu[i, j]` is the estimated difference in expectation of $A(x)$ between states `i` and `j`.
    pub mu: Array2<f64>,
    /// `sigma[i, j]` is the uncertainty estimate at one standard deviation for the difference in $A$ beteen `i` and `j`
    pub sigma: Array2<f64>,
}

/// Free energy differences decomposed into enthalpy and entropy
#[derive(Debug)]
pub struct EntropyEnthalpy {
    /// `delta_f[i, j]` is the dimensionless free energy difference $f_j - f_i$
    pub delta_f: Array2<f64>,
    /// `d_delta_f[i, j]` is the uncertainty in the dimensionless free energy difference $f_j - f_i$
    pub d_delta_f: Array2<f64>,
    /// `delta_u[i, j]` is the reduced potential energy difference $u_j - u_i$
    pub delta_u: Array2<f64>,
    /// `d_delta_u[i, j]` is the uncertainty in the reduced potential energy difference $u_j - u_i$
    pub d_delta_u: Array2<f64>,
    /// `delta_s[i, j]` is the reduced entropy difference $S_j/k_b - S_i/k_b$
    pub delta_s: Array2<f64>,
    /// `d_delta_s[i, j]` is the uncertainty in the reduced entropy difference $S_j/k_b - S_i/k_b$
    pub d_delta_s: Array2<f64>,
}
