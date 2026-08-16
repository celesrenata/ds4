# TensorOps Feature Tier Compile Probes

Minimal Metal source files that test SDK availability of each TensorOps
feature tier. Run on macOS with:

```bash
xcrun metal -c probe_tensorops_tier1.metal -o /dev/null && echo "Tier 1: PASS"
xcrun metal -c probe_tensorops_tier2.metal -o /dev/null && echo "Tier 2: PASS"
xcrun metal -c probe_tensorops_tier3.metal -o /dev/null && echo "Tier 3: PASS"
xcrun metal -c probe_tensorops_tier4.metal -o /dev/null && echo "Tier 4: PASS"
```

## Tiers

| File | Tier | Feature | Min SDK |
|------|------|---------|---------|
| `probe_tensorops_tier1.metal` | 1 | Basic tensor/matmul2d/cooperative destination | macOS 26.0 |
| `probe_tensorops_tier2.metal` | 2 | Cooperative tensor as matmul input | macOS 26.3 |
| `probe_tensorops_tier3.metal` | 3 | Native int4/int8 tensor element types | macOS 26.4 |
| `probe_tensorops_tier4.metal` | 4 | Low-bit float + E8M0 scale planes (MX) | macOS 27.0 |

## Status

- Tier 1 is already probed at runtime by `ds4_gpu_compile_tensor_probe()` in `ds4_metal.m`.
- Tiers 2–4 are stubs awaiting macOS hardware confirmation. Adjust types to
  match actual SDK symbols when the target toolchain is available.

## Notes

- Tier 3 and 4 probes contain placeholder/commented code because the exact
  type names are not confirmed without the installed SDK headers.
- Once probes compile on target hardware, their key declarations should be
  incorporated into `ds4_gpu_detect_metal4_features()` as additional
  runtime probes (Tasks 2.1 and 2.2).
