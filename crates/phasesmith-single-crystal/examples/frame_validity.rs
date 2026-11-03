//! Run shadow estimation and full-exposure DAC masking without Python or file I/O.
use phasesmith_single_crystal::spots::frame_validity::{
    self, DacOptions, Mask, SeatOptions, Shape,
};
use phasesmith_single_crystal::spots::reciprocal::Geometry;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shape = Shape {
        height: 192,
        width: 256,
        block: 16,
    };
    let [rows, columns] = shape.coarse()?;
    let cells = rows * columns;
    let mut scan = Vec::new();
    let mut available = Vec::new();
    // Replace these synthetic counts with frames from any Rust reader.
    for frame in 0..20 {
        let counts: Vec<f64> = (0..shape.height * shape.width)
            .map(|i| {
                if frame < 5 && i % shape.width >= 128 {
                    15.
                } else {
                    80.
                }
            })
            .collect();
        let snapshot = frame_validity::coarsen(&counts, &vec![false; counts.len()], shape)?;
        scan.extend(snapshot.values);
        available.push(snapshot.available);
    }
    let estimate = frame_validity::estimate(&scan, &available, shape, SeatOptions::default())?;
    let first = Mask {
        shape,
        excluded: estimate.excluded[..cells].to_vec(),
    };
    let geometry = Geometry {
        origin: [-0.0128, -0.0096, 0.2],
        fast: [0.0001, 0., 0.],
        slow: [0., 0.0001, 0.],
        beam: [0., 0., 1.],
        axis: [1., 0., 0.],
        wavelength: 0.28457,
    };
    let aperture = frame_validity::dac_mask(
        geometry,
        [-34., -33.5],
        shape,
        DacOptions {
            axis_at_zero: [0., 0., 1.],
            half_opening_degrees: 35.,
        },
    )?;
    let mut combined = first.expand()?;
    for (excluded, geometric) in combined.iter_mut().zip(aperture.expand()?) {
        *excluded |= geometric;
    }
    println!(
        "reference frames: {:?}; unavailable pixels in first frame: {}",
        estimate.reference_indices,
        combined.iter().filter(|v| **v).count()
    );
    Ok(())
}
