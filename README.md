# mbar-rs
Rust bindings for pymbar. Requires nightly for const generics.

```rust
use mbar_rs::*;

fn main() {
    // Construct the MBar struct
    let mbar = MBar::builder()
        // The potential energies of each snapshot, evaluated at all states
        .u_kn([
            [1.4, 2.3, 3.7, 4.1, 7.7, 9.1],
            [-1.6, -2.3, 9.7, 34.1, 27.7, 19.1],
            [4.4, 7.3, 9.7, 8.1, 4.7, 3.1],
        ])
        // The number of snapshots from each state
        .n_k([2, 2, 2])
        .build()
        .unwrap();

    // And then compute a PMF
    let pmf = mbar
        .compute_pmf(
            // The potential energies at the target state
            [1.4, 2.3, 3.7, 4.1, 7.7, 9.1],
            // The bins of each snapshot
            [0, 1, 2, 0, 2, 1],
            // The number of bins
            3,
            // How to compute the uncertainties
            PmfUncertainties::FromLowest,
        )
        // If there's a Python error, display it nicely
        // This shouldn't happen, as the Rust API should catch everything (mostly at compile time!)
        .map_err(|e| {
            println!("{}", e);
            e
        });

    // Print out the pmf!
    println!("{:?}", pmf)
}
```
