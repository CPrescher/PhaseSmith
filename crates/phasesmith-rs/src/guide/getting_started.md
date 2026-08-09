# Getting started

Add the facade crate to an application:

```text
cargo add phasesmith
```

The facade deliberately keeps component namespaces visible. This makes the
owner of a type apparent and avoids a very large, collision-prone prelude.

## Calculate profile values and derivatives

[`crate::core::accumulate_batch`] performs one fused traversal and returns a
calculated pattern plus analytical derivatives. Local derivative columns are
ordered as intensity, position, FWHM, and eta.

```
use phasesmith::core::{GridView, PeakBatchView, SupportPolicy, accumulate_batch};

let x = [23.9, 24.0, 24.1];
let positions = [24.0];
let intensities = [100.0];
let fwhms = [0.1];
let etas = [0.4];

let result = accumulate_batch(
    GridView::new(&x)?,
    PeakBatchView::new(&positions, &intensities, &fwhms, &etas)?,
    SupportPolicy::FwhmMultiple(20.0),
)?;

assert_eq!(result.sample_count, 3);
assert_eq!(result.derivatives.local.parameter_count, 4);
assert!(result.y[1] > result.y[0]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Derive unit-cell geometry

Unit-cell lengths are ångströms and angles are degrees. Reciprocal vectors do
not include a `2π` factor.

```
use phasesmith::crystallography::UnitCell;

let cell = UnitCell {
    a_angstrom: 5.0,
    b_angstrom: 5.0,
    c_angstrom: 5.0,
    alpha_deg: 90.0,
    beta_deg: 90.0,
    gamma_deg: 90.0,
};
let geometry = cell.geometry()?;
let (d_angstrom, derivatives) =
    geometry.d_spacing_and_derivatives([1, 0, 0])?;

assert!((d_angstrom - 5.0).abs() < 1.0e-12);
assert_eq!(derivatives.len(), 6);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Parse observed powder data

Native readers enforce explicit allocation limits and return owned
[`crate::model::PatternRecord`] values.

```
use phasesmith::io::{
    PowderFormat, PowderReadLimits, parse_powder_text,
};

let data = parse_powder_text(
    "20.0 100.0 2.0\n20.1 120.0 2.5\n",
    PowderFormat::Columns,
    1,
    PowderReadLimits::default(),
)?;

assert_eq!(data.pattern.sample_count(), 2);
assert_eq!(data.pattern.observed_y.as_deref(), Some(&[100.0, 120.0][..]));
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Next steps

- Calculate from a crystal structure: [`crate::engine`].
- Extract intensities or refine a structure: [`crate::guide::workflows`].
- Save application state: [`crate::persistence`].
- Integrate background work into a GUI: [`crate::guide::application_hosts`].
