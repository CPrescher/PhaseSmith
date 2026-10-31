# Third-party notices

PhaseSmith is independently licensed under the MIT License. The following
notice covers compatible behavior derived from separately licensed software.

## Published DAFi algorithm

The experimental **Row search (DAFi-based)** method independently implements
equidistant-row grouping described by Andrey Aslandukov, Matvii Aslandukov,
Natalia Dubrovinskaia and Leonid Dubrovinsky, “Domain Auto Finder (DAFi)
program: the analysis of single-crystal X-ray diffraction data from
polycrystalline samples”, *Journal of Applied Crystallography* **55** (2022),
1383–1391, DOI `10.1107/S1600576722008081`. The paper is distributed under
CC BY 4.0. No original DAFi implementation source or executable is incorporated
or required. This is an independent implementation with documented numerical
choices, not official DAFi or a claim of exact original-software equivalence.

## GSAS-II black-box validation

PhaseSmith's optional oracle fixtures contain plain numerical outputs generated
with the exact GSAS-II revision recorded in `oracle/PINNED_GSASII.json`.
PhaseSmith does not redistribute GSAS-II source or binaries and does not use it
at runtime. GSAS-II is copyright UChicago Argonne, LLC and is distributed under
its own [open-source license](https://github.com/AdvancedPhotonSource/GSAS-II/blob/c0bc79b259cdf0065480b5fbd57674ddf12c4a23/LICENSE).
The upstream-requested acknowledgment is: “This product includes software
produced by UChicago Argonne, LLC under Contract No. DE-AC02-06CH11357 with the
Department of Energy.” The primary scientific citation is B. H. Toby and
R. B. Von Dreele, *Journal of Applied Crystallography* **46** (2013), 544–549,
DOI `10.1107/S0021889813003531`.

## xypattern Smooth Bruckner background algorithm

The `phasesmith.background` Bruckner smoother reproduces the observable algorithm
of xypattern revision `6e4574d75d2d6fcefc633f9fbecc27b8f1bcd817`.

MIT License

Copyright (c) 2025 Clemens Prescher

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Native persistence dependencies

The native JSON+NPZ persistence codec uses `zip` 5.1.1 (MIT), `flate2` 1.1
(`MIT OR Apache-2.0`), and `sha2` 0.10.9 (`MIT OR Apache-2.0`). PhaseSmith uses
the MIT option for the dual-licensed crates.

Copyright (c) 2014 Mathijs van de Nes (`zip`)

Copyright (c) 2014-2026 Alex Crichton (`flate2`)

Copyright (c) 2006-2009 Graydon Hoare

Copyright (c) 2009-2013 Mozilla Foundation

Copyright (c) 2016 Artyom Pavlov (`sha2`)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Moyo 0.15.0

The compiled native space-group database and conversion code use Moyo 0.15.0,
published by Kohei Shinohara and contributors under `MIT OR Apache-2.0`.
PhaseSmith uses the MIT option:

MIT License

Copyright (c) 2023 Kohei Shinohara

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## RustFFT 6.4.1 (experimental structure solution)

The opt-in `phasesmith-solution` crate uses RustFFT 6.4.1, licensed under
`MIT OR Apache-2.0`; PhaseSmith uses the MIT option. Its FFT implementation
is a dependency, separate from the independently implemented crystallographic
phasing equations. Upstream: https://github.com/ejmahler/RustFFT

Copyright (c) 2015 The RustFFT Developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Dioptas frame validity and seat-shadow estimator

`phasesmith.experimental.frame_validity` adapts the mask representation and
original-count illumination-ratio estimator from Dioptas
`dioptas/model/frame_validity.py`, revision
`651c27f98108fb19f60db9a8cb227755b2f9d6a2`. Application readers, display transforms,
resource settings and UI/cache dependencies are not included. The source is
MIT-licensed:

MIT License

Copyright (c) 2014-2019 GSECARS, University of Chicago, USA
Copyright (c) 2015-2018 Institute for Geology and Mineralogy, University of Cologne, Germany
Copyright (c) 2019-2020 DESY, Hamburg, Germany
Copyright (c) 2021-2026 Clemens Prescher

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
