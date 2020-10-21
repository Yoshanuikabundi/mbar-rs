use mbar_rs::*;

fn main() {
    let mbar = MBar::builder()
        .u_kn([
            [1.4, 2.3, 3.7, 4.1, 7.7, 9.1],
            [-1.6, -2.3, 9.7, 34.1, 27.7, 19.1],
            [4.4, 7.3, 9.7, 8.1, 4.7, 3.1],
        ])
        .n_k([2, 2, 2])
        .build()
        .unwrap();

    let pmf = mbar
        .compute_pmf(
            [1.4, 2.3, 3.7, 4.1, 7.7, 9.1],
            [0, 1, 2, 0, 2, 1],
            3,
            PmfUncertainties::FromLowest,
        )
        .map_err(|e| {
            println!("{}", e);
            e
        });

    println!("{:?}", pmf)
}
