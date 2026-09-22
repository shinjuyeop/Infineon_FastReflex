# Simulation images

[Project overview](../../README.md#simulation-preview) · [Recording guide](../simulation.md#recording-presentation-videos)

These are lossless PNG frame extractions from existing MuJoCo demo recordings,
selected for documentation on 2026-09-22. No generated illustration, reconstructed
model output, or board screenshot is used. The source videos remain local under
`simulation/outputs/`; the two reviewed stills total approximately 742 KiB.

| Image | Source video | Run / split | Video position | Visible simulation time |
|---|---|---|---|---|
| [Ice](ice_hazard.png) | `ice_hazard_demo.mp4` | `uhr_ice_h_c20` / VALIDATION | 5.400 s | 2.701 s |
| [Sand](sand_hazard.png) | `sand_hazard_demo.mp4` | `uhr_sand_h_c24` / VALIDATION | 5.800 s | 2.901 s |

Both recordings use the research Float Hazard/Terrain demo at 0.5x playback,
1920 × 1080, 60 fps. The existing recording commands and selection rules are in
the [simulation guide](../simulation.md#recording-presentation-videos), present at
source commit `434a407`. The Sand recording uses the documented presentation case
`c24`; the default representative Sand replay remains `c20`.

The selected frames show a recorded Hazard alert and a later terrain estimate.
`DANGER` retains alert history for display; it is not a recovery controller or
proof that the detector waits for Terrain. Physical reference labels are
simulator diagnostics, not model inputs. These images do not establish E84
inference, real-robot validation, or generalization performance. No new experiment
or HOLDOUT replay was run to create them.

## Extract the same frames

From the research repository root with the original local videos available:

```bash
ffmpeg -n -hide_banner -loglevel error \
  -ss 5.4 -i simulation/outputs/ice_hazard_demo.mp4 \
  -frames:v 1 -update 1 docs/assets/ice_hazard.png
ffmpeg -n -hide_banner -loglevel error \
  -ss 5.8 -i simulation/outputs/sand_hazard_demo.mp4 \
  -frames:v 1 -update 1 docs/assets/sand_hazard.png
```

`-n` refuses to overwrite the checked-in stills. To inspect a new extraction,
choose a temporary output path. FFmpeg encoder versions may change PNG bytes;
the source-video hashes and timestamps identify the selected content.

## SHA-256 provenance

| File | SHA-256 |
|---|---|
| `ice_hazard_demo.mp4` | `aa1f120e77ffd45486f33eed1ba7851fc5932817fec6a5f9460a34c7c4f80bb5` |
| `sand_hazard_demo.mp4` | `1ac94614b7f9a7fd4fece639761797e84857c46aa69c8c6ad129b2907fb3e754` |
| `ice_hazard.png` | `c6e21c5fd55923547058c15f189e38ac0d326558673e082e5eabe75f293f6ff8` |
| `sand_hazard.png` | `05351ecc36d9221fbdf9bba820bdf22e7f69b5435f1e62173e2c5813a3c05d25` |

The G1 geometry and meshes retain their [upstream provenance and license](../../src/fastreflex/simulation/assets/unitree_g1/NOTICE.md).
