"""Build an isolated copy of our own Rust row kernel with one GPU hook.

No production source or workspace manifest is edited. Exact patch anchors and
source SHA256 are recorded so the numerical baseline is reviewable.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    source_path = root / "crates/phasesmith-single-crystal/src/row_search/mod.rs"
    source = source_path.read_text()
    original = source

    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, old
        source = source.replace(old, new)

    replace(
        "        let partial = std::thread::scope(|scope| {",
        "        let gpu_seeds = if groups.is_empty() {\n"
        "            gpu_initial_preparations(&p, &ranked[..count], o, start, cancel)?\n"
        "        } else { Vec::new() };\n"
        "        let partial = std::thread::scope(|scope| {",
    )
    replace(
        "                let alive = &alive;\n                handles.push(scope.spawn(move || {",
        "                let alive = &alive;\n"
        "                let gpu_seeds = &gpu_seeds;\n"
        "                handles.push(scope.spawn(move || {",
    )
    anchor = """                            let rows = prepare_direction_scratch(
                                p,
                                ids,
                                decode(ranked[rank].0, o.direction_epsilon),
                                o,
                                &mut scratch,
                            );"""
    replace(
        anchor,
        """                            let seed = gpu_seeds.get(rank)
                                .and_then(|x| x.lock().unwrap().take());
                            let rows = if let Some(seed) = seed {
                                seed
                            } else {
                                prepare_direction_scratch(p,ids,
                                    decode(ranked[rank].0,o.direction_epsilon),o,&mut scratch)
                            };""",
    )
    extra = (Path(__file__).parent / "rows_bridge_extra.rs").read_text()
    out = a.output / "bridge"
    (out / "src").mkdir(parents=True, exist_ok=True)
    (out / "src/lib.rs").write_text(source + "\n" + extra)
    (out / "Cargo.toml").write_text("""[package]
name = "phasesmith-row-gpu-experiment"
version = "0.0.0"
edition = "2024"
[workspace]
[lib]
crate-type = ["cdylib"]
[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
""")
    subprocess.run(
        ["cargo", "build", "--release", "--offline", "--manifest-path", str(out / "Cargo.toml")],
        check=True,
    )
    (a.output / "bridge-provenance.json").write_text(
        json.dumps(
            dict(
                source=str(source_path),
                source_sha256=hashlib.sha256(original.encode()).hexdigest(),
                patch=(
                    "Initial full-population preparations only; three exact patch anchors; "
                    "numerical grouping and CPU baseline inherited unchanged."
                ),
                extra_sha256=hashlib.sha256(extra.encode()).hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
