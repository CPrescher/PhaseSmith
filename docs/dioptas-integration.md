# Dioptas integration boundary

`rietveld.integrations.dioptas` is a thin, optional NumPy interface. It imports
without Dioptas installed and contains no Qt widgets, signals, project objects,
or GUI event handling.

`DioptasPatternData` accepts degree-two-theta `x`, observed intensity, optional
background and one-sigma uncertainty, and either an included or excluded
boolean mask. Internally, masks always use `True = included`. Inputs are copied,
validated, and converted to the ordinary `PowderPattern` model.

`calculate` returns total calculated intensity, explicitly separate background,
`observed - calculated` difference, included mask, labeled phase curves, and
plain diagnostics. Both boundary records expose `x_unit = "degree_2theta"` and
`intensity_unit = "arbitrary_intensity"`; adapters must perform any conversion
before constructing `DioptasPatternData`. `refine_lebail` returns that display
record plus the complete typed `LeBailResult`.

```python
from rietveld.integrations import dioptas

source = dioptas.DioptasPatternData(
    two_theta,
    measured,
    background_y=background,
    uncertainty_y=sigma,
    excluded_mask=gui_excluded_mask,
    metadata={"source": "Dioptas"},
)
display, result = dioptas.refine_lebail(source, instrument, phases)
plot.set_data(display.x_deg, display.calculated_y)
```

For application-owned glue, `DioptasConsumer` has only `read_pattern()` and
`publish_result(result)`. Tests use a fake consumer to lock down dtypes, shapes,
labels, mask conversion, background separation, and errors. Dioptas itself
never becomes a package dependency.

## Progress, cancellation, threads, and the GIL

Calculation progress is reported immediately before and after one native batch.
Le Bail progress is reported after complete iterations with Rwp and intensity
change. Cancellation is polled only at those boundaries; a native accumulator
is never interrupted with partially written arrays. Calculation cancellation
raises `OperationCancelled`; Le Bail returns a valid checkpoint and
`TerminationReason.CANCELLED`.

Callbacks execute synchronously on the thread that called the library. GUI glue
must marshal `publish_result` and widget changes back to the GUI-owned thread.
At present, PyO3 accumulation calls retain the Python GIL while Rust evaluates
the batch. For reliably responsive Python/Qt applications, run calculation or
refinement in a worker process and transfer the plain NumPy/result records back
to the GUI process. A Python worker thread alone is not yet a guarantee of GUI
responsiveness. Releasing the GIL safely is a future binding optimization and
will not change the progress/cancellation contract.
