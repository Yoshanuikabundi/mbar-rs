use anyhow::Result;
use mbar_rs::*;
use ndarray::array;

fn main() -> Result<()> {
    let mbar = MBar::builder()
        .u_kn(array![
            [1.4, 2.3, 3.7, 4.1, 7.7, 9.1],
            [-1.6, -2.3, 9.7, 34.1, 27.7, 19.1],
            [4.4, 7.3, 9.7, 8.1, 4.7, 3.1],
        ])
        .n_k(array![2, 2, 2])
        .build_verbose()?;

    let pmf = mbar.compute_pmf(
        &[1.4, 2.3, 3.7, 4.1, 7.7, 9.1],
        &[0, 1, 2, 0, 2, 1],
        3,
        PmfUncertainties::FromLowest,
    )?;

    println!("{:?}", pmf);

    Ok(())
}
