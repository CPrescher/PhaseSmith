//! 0.8.x compatibility of the single-crystal grain record, now re-exported
//! from `phasesmith_single_crystal::grains`: callers outside the crate can
//! still build it with a struct literal and match it exhaustively.
#![cfg(feature = "single-crystal")]
use phasesmith_workflows::single_crystal::Grain;

#[test]
fn grain_is_constructible_and_matchable_outside_the_crate() {
    let grain = Grain {
        direct: [[5., 0., 0.], [0., 6., 0.], [0., 0., 7.]],
        hkl: vec![[1, 0, 0]],
        indexed: vec![true],
        training: 1,
        withheld: 0,
        accepted: false,
        null_counts: vec![0; 64],
    };
    let Grain {
        direct,
        hkl,
        indexed,
        training,
        withheld,
        accepted,
        null_counts,
    } = grain.clone();
    assert_eq!(direct[1][1].to_bits(), 6_f64.to_bits());
    assert_eq!(
        (hkl.len(), indexed, training, withheld, accepted),
        (1, vec![true], 1, 0, false)
    );
    assert_eq!(null_counts.len(), 64);
    assert!(format!("{grain:?}").starts_with("Grain"));
}
