# mbar-rs

[![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/yoshanuikabundi/mbar-rs?label=tag&logo=github&sort=semver)](https://github.com/Yoshanuikabundi/mbar-rs)
[![Crates.io](https://img.shields.io/crates/v/mbar-rs?label=crates.io&logo=rust)](https://crates.io/crates/mbar-rs)
[![lib.rs](https://img.shields.io/crates/v/mbar-rs?label=lib.rs&logo=rust)](https://lib.rs/crates/mbar-rs)
[![Latest Documentation](https://docs.rs/mbar-rs/badge.svg)](https://docs.rs/mbar-rs)
[![MIT License](https://img.shields.io/github/license/yoshanuikabundi/mbar-rs)](https://github.com/Yoshanuikabundi/mbar-rs/blob/main/LICENSE)

Safe and Rusty bindings for [pymbar](https://github.com/choderalab/pymbar). Thanks to Kyle A. Beauchamp, John D. Chodera, Levi N. Naden and Michael R. Shirts for their work on the underlying Python library, which must be available on your machine for this to work. I've also cribbed shamelessly from their documentation.

# Example

```rust
use mbar_rs::*;

fn main() {
    // Construct the MBar struct
    let mbar = MBar::builder()
        // The potential energies of each snapshot, evaluated at all states
        .u_kn(array![
            [1.4, 2.3, 3.7, 4.1, 7.7, 9.1],
            [-1.6, -2.3, 9.7, 34.1, 27.7, 19.1],
            [4.4, 7.3, 9.7, 8.1, 4.7, 3.1],
        ])
        // The number of snapshots from each state
        .n_k(array![2, 2, 2])
        .build()
        .unwrap();

    // And then compute a PMF
    let pmf = mbar
        .compute_pmf(
            // The potential energies at the target state
            &[1.4, 2.3, 3.7, 4.1, 7.7, 9.1],
            // The bins of each snapshot
            &[0, 1, 2, 0, 2, 1],
            // The number of bins
            3,
            // How to compute the uncertainties
            PmfUncertainties::FromLowest,
        )
        // If there's a Python error, display it nicely
        // This shouldn't happen, as the Rust API should catch everything (mostly at compile time!)
        .map_err(|e| e.print())
        .unwrap();

    // Print out the pmf!
    println!("{:?}", pmf)
}
```


# References

* Please cite the original MBAR paper:

[1] Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states. J. Chem. Phys. 129:124105 (2008).  [DOI](http://dx.doi.org/10.1063/1.2978177)

* Some timeseries algorithms can be found in the following reference:

[2] Chodera JD, Swope WC, Pitera JW, Seok C, and Dill KA. Use of the weighted histogram analysis method for the analysis of simulated and parallel tempering simulations. J. Chem. Theor. Comput. 3(1):26-41 (2007).  [DOI](http://dx.doi.org/10.1021/ct0502864)

* The automatic equilibration detection method provided in `pymbar.timeseries.detectEquilibration()` is described here:

[3] Chodera JD. A simple method for automated equilibration detection in molecular simulations. J. Chem. Theor. Comput. 12:1799, 2016.  [DOI](http://dx.doi.org/10.1021/acs.jctc.5b00784)
