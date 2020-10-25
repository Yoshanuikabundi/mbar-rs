use crate::errors::*;
use ndarray::{array, Array1, Array2};
use numpy::{PyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::convert::TryFrom;

/// Test cases using harmonic oscillators.
///
/// # Examples
///
/// Generate energy samples with default parameters.
///
/// ```
/// use mbar_rs::testsystems::*;
/// use ndarray::array;
///
/// let testcase = HarmonicOscillator::default();
/// let Sample {x_n, u_kn, n_k, s_n} = testcase.sample(array![10, 20, 30, 40, 50]).unwrap();
/// ```
///
/// Retrieve analytical properties.
/// >>> analytical_means = testcase.analytical_means()
/// >>> analytical_variances = testcase.analytical_variances()
/// >>> analytical_standard_deviations = testcase.analytical_standard_deviations()
/// >>> analytical_free_energies = testcase.analytical_free_energies()
/// >>> analytical_x_squared = testcase.analytical_observable('position^2')
/// Generate energy samples with default parameters in one line.
/// >>> (x_kn, u_kln, N_k, s_n) = HarmonicOscillatorsTestCase().sample()
/// Generate energy samples with specified parameters.
/// >>> testcase = HarmonicOscillatorsTestCase(O_k=[0, 1, 2, 3, 4], K_k=[1, 2, 4, 8, 16])
/// >>> (x_kn, u_kln, N_k, s_n) = testcase.sample(N_k=[10, 20, 30, 40, 50])
/// Test sampling in different output modes.
/// >>> (x_kn, u_kln, N_k) = testcase.sample(N_k=[10, 20, 30, 40, 50], mode='u_kln')
/// >>> (x_n, u_kn, N_k, s_n) = testcase.sample(N_k=[10, 20, 30, 40, 50], mode='u_kn')
#[derive(Builder, Debug)]
#[builder(build_fn(validate = "Self::validate", name = "build_inner", private))]
pub struct HarmonicOscillator {
    /// Offset parameters for each state.
    #[builder(default = "array![0.0, 1.0, 2.0, 3.0, 4.0]")]
    o_k: Array1<f64>,

    /// Force constants for each state.
    #[builder(default = "array![1.0, 2.0, 4.0, 8.0, 16.0]")]
    k_k: Array1<f64>,

    /// Inverse temperature
    #[builder(default = "1.0")]
    beta: f64,

    /// Pointer to the HarmonicOscillatorsTestCase object on Python's heap
    ///
    /// After Self is built, this should always be a valid pointer to an HarmonicOscillatorsTestCase
    /// object. This is enforced by it being a private field and HarmonicOscillator structs only
    /// being constructable via the builder pattern.
    #[builder(setter(skip), default = "Python::with_gil(|py| py.None())")]
    pyobj: PyObject,
}

impl HarmonicOscillatorBuilder {
    fn validate(&self) -> std::result::Result<(), String> {
        if let (Some(o_k), Some(k_k)) = (&self.o_k, &self.k_k) {
            if o_k.len() != k_k.len() {
                return Err(format!(
                    "o_k and k_k must have equal lengths (not {} and {})",
                    o_k.len(),
                    k_k.len()
                ));
            }
        }

        Ok(())
    }

    /// Build and initialise the MBAR implementation
    pub fn build(&self) -> Result<HarmonicOscillator> {
        self.build_inner()?.init()
    }
}

impl HarmonicOscillator {
    /// Initialise the MBAR in Python; called by build methods
    fn init(mut self) -> Result<Self> {
        Python::with_gil(|py| {
            let pyobj = py
                .import("pymbar.testsystems.harmonic_oscillators")?
                .get("HarmonicOscillatorsTestCase")?;

            let kwargs: &PyDict = PyDict::new(py);
            kwargs.set_item("O_k", PyArray::from_array(py, &self.o_k))?;
            kwargs.set_item("K_k", PyArray::from_array(py, &self.k_k))?;
            kwargs.set_item("beta", self.beta)?;

            self.pyobj = PyAny::call(pyobj, (), Some(kwargs))?.to_object(py);
            Ok(self)
        })
    }

    /// Get a new builder for the `MBar` struct. `MBar` can only be constructed via the builder.
    pub fn builder() -> HarmonicOscillatorBuilder {
        HarmonicOscillatorBuilder::default()
    }

    /// Draw samples from the distribution with a random seed
    pub fn sample(&self, n_k: Array1<usize>) -> Result<Sample> {
        self.sample_inner(n_k, None)
    }

    /// Draw samples from the distribution with a specified seed
    pub fn sample_with_seed(&self, n_k: Array1<usize>, seed: u128) -> Result<Sample> {
        self.sample_inner(n_k, Some(seed))
    }

    fn sample_inner(&self, n_k: Array1<usize>, seed: Option<u128>) -> Result<Sample> {
        Python::with_gil(|py| {
            let kwargs: &PyDict = PyDict::new(py);
            kwargs.set_item("N_k", PyArray::from_array(py, &n_k))?;
            kwargs.set_item("seed", seed)?;
            kwargs.set_item("mode", "u_kn")?;

            let pytup = self
                .pyobj
                .as_ref(py)
                .call_method("sample", (), Some(kwargs))?;

            Ok(Sample::try_from(pytup)?)
        })
    }
}

impl Default for HarmonicOscillator {
    fn default() -> Self {
        Self::builder()
            .build()
            .expect("HarmonicOscillator should not fail with default params")
    }
}

/// A sample from a test case
#[derive(Debug)]
pub struct Sample {
    /// `x_n[n]` is sample n (in concatenated indexing)
    pub x_n: Array1<f64>,
    /// `u_kn[k,n]` is the reduced potential of sample n (in concatenated indexing) evaluated at state k.
    pub u_kn: Array2<f64>,
    /// `n_k[k]` is the number of samples generated from state k
    pub n_k: Array1<usize>,
    /// `s_n[k]` is the state of origin of `x_n[k]`
    pub s_n: Array1<usize>,
}

impl TryFrom<&PyAny> for Sample {
    type Error = MBarError;

    fn try_from(value: &PyAny) -> Result<Self> {
        let x_n = value
            .get_item(0)?
            .extract::<&PyArray1<f64>>()?
            .to_owned_array();

        let u_kn = value
            .get_item(1)?
            .extract::<&PyArray2<f64>>()?
            .to_owned_array();

        let n_k = value
            .get_item(2)?
            .extract::<&PyArray1<i32>>()?
            .iter()?
            .map(|&mut i| usize::try_from(i))
            .collect::<std::result::Result<Array1<usize>, std::num::TryFromIntError>>()?;

        let s_n = value
            .get_item(3)?
            .extract::<&PyArray1<i64>>()?
            .iter()?
            .map(|&mut i| usize::try_from(i))
            .collect::<std::result::Result<Array1<usize>, std::num::TryFromIntError>>()?;

        Ok(Self {
            x_n,
            u_kn,
            n_k,
            s_n,
        })
    }
}
