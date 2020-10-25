use anyhow::Result;
use itertools::Itertools;
use mbar_rs::mbar::*;
use mbar_rs::testsystems::*;
use ndarray::{array, s, Array1, Axis};

/// Produce a sorted copy of an array
fn sorted(s: &Array1<f64>) -> Array1<f64> {
    let mut new = s.clone();
    new.as_slice_mut()
        .unwrap()
        .sort_by(|a, b| a.partial_cmp(b).expect("Tried to compare a NaN"));
    new
}

/// Return the indices of the bins to which each value in input array belongs.
fn digitise<T: PartialOrd>(x_n: &[T], bin_edges: &[T]) -> Vec<usize> {
    x_n.iter()
        .map(|x| {
            bin_edges.iter().enumerate().find_map(
                |(i, left)| {
                    if x < left {
                        Some(i - 1)
                    } else {
                        None
                    }
                },
            )
        })
        .map(|opt| opt.expect("Value didn't get a bin"))
        .collect()
}

fn main() -> Result<()> {
    // Generate some sample data
    let testcase = HarmonicOscillator::default();

    let Sample {
        x_n,
        u_kn,
        n_k,
        s_n,
    } = dbg!(testcase
        .sample(array![100, 200, 300, 400, 500])
        .map_err(|e| e.print())
        .unwrap());

    println!("{:?}", n_k);

    // Build the mbar struct
    let mbar = MBar::builder()
        .u_kn(u_kn.clone())
        .n_k(n_k)
        .x_kindices(s_n)
        .build_verbose()?;

    // Divide x_n into equally populated bins
    let n_bins = 10;
    let n_tot = x_n.len();
    let x_n_sorted = sorted(&x_n);
    let bin_size = n_tot / n_bins;
    let mut bin_edges = x_n_sorted.slice(s![..;bin_size]).to_vec();
    bin_edges.push(x_n_sorted[n_tot - 1] + 0.001);

    let bin_n = digitise(&x_n.as_slice().unwrap(), &bin_edges);

    let bin_widths: Array1<f64> = bin_edges
        .iter()
        .tuple_windows()
        .map(|(left, right)| (right - left).abs())
        .collect();

    // Compute the PMF
    let pmf = mbar.compute_pmf(
        &u_kn.index_axis(Axis(0), 0).to_slice().unwrap(),
        &dbg!(bin_n),
        n_bins,
        PmfUncertainties::FromLowest,
    )?;

    println!("{:?}", &pmf);

    // Correct for unequally spaced bins to get a PMF on uniform measure
    let f_i_corrected: Vec<_> = pmf
        .f_i
        .iter()
        .zip(bin_widths.iter())
        .map(|(f, w)| f - w.ln())
        .collect();

    println!("{:?}", &f_i_corrected);

    println!("{:?}", bin_edges);

    Ok(())
}
