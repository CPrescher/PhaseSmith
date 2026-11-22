# PhaseSmith branding

Two candidate marks live here until a final decision is made: the **lit phase
scale** (the default file names below) and the **plain three-plane mark**
(files suffixed `-n1`), which is the same geometry reduced to three flat,
single-hue planes with no gradients.

The phase-scale PhaseSmith mark is a stack of four inclined lattice planes, stacked along a
normal that points to the upper left. Each plane advances in phase, and the
phase is encoded as color — deep teal at the back through teal and mint to hot
amber in front, the way structure-factor phases are drawn on a color scale.
Depth comes from lighting: every plane is lit from the direction of the plane
normal, and the stack fades into the background with distance (the fade is
opacity-based, so it adapts to light and dark backgrounds alike). The amber
front plane carries the "smith": freshly forged, still hot.

## Files

| File | Use |
| --- | --- |
| `logo.svg` | Primary mark. Transparent background, works on light and dark. Use at 32 px and above. |
| `logo-solid.svg` | Small-size cut: gradients flattened to solid phase colors. Use below 32 px when not using `favicon.svg`. |
| `logo-mono.svg` | Single-color version (`currentColor` with opacity steps). For plots, papers, stamps, and any one-ink context. |
| `viewer-icon.svg` | PhaseSmith Viewer app tile: the mark at 82% on the dark rounded square. |
| `favicon.svg` | Browser-tab icon: solid cut on the dark rounded square. |
| `logo-n1.svg` | Plain candidate: three flat planes, teal/mint, no gradients. One file for all sizes. |
| `logo-n1-mono.svg` | Plain candidate in `currentColor` with opacity steps. |
| `viewer-icon-n1.svg` | Plain candidate on the dark Viewer tile. |
| `favicon-n1.svg` | Plain candidate as browser-tab icon. |

## Colors

Phase scale (back to front):

| Role | Hex |
| --- | --- |
| Phase 0 (back) | `#0A5D4F` → `#2DB49B` (lit edge) |
| Phase 1 | `#189380` → `#4ADEB4` |
| Phase 2 | `#34D399` → `#A7F3D0` |
| Phase 3 (front, forge-hot) | `#F59E0B` → `#FDE68A` |

Supporting colors: ink `#0A110E` (tile background), accent teal for text
`#0E8F7E` (AA on white), light surface `#F1F5F3`.

## Wordmark

Set "PhaseSmith" in Space Grotesk: "Phase" at weight 500 in near-black
(`#15201C`), "Smith" at weight 700 in teal (`#0E8F7E`), letter-spacing −0.015em.
The Viewer is badged with a small uppercase "VIEWER" chip (`#0A6B5E` on
`#E3F3ED`), never by changing the wordmark itself.

## Rules of thumb

- Never redraw the planes as a 2×2 grid or stack them vertically (Windows and
  Buffer territory, respectively); the diagonal upper-left normal is part of
  the identity.
- The gradient mark and the solid cut are the same geometry; pick by size, do
  not mix within one surface.
- The phase ramp (teal → amber) is also the preferred sequential colormap
  accent for Viewer UI highlights.
