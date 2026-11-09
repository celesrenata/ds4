# GLM support is disabled by default

This tree focuses on Apple Metal DeepSeek V4.1. GLM 5.2 / GLM 5.3 code is
kept in the source but is switched off.

## What is off

- Model dispatch: `config_validate_model()` in `ds4.c` refuses GGUFs whose
  `general.architecture` is `glm-dsa` or `glm5-next` with a clear error.
  The GLM config validators and the `DS4_SHAPE_GLM52` shape (plus two helpers
  only they used) are compiled out.
- Build targets: `test-glm53-kda`, `test-glm-attention`, and the
  `tests/test_glm53_*` objects/binaries are only defined when GLM is enabled.
  The ROCm GLM targets and all CUDA/ROCm files are untouched.

Nothing was deleted. The remaining GLM code paths are unreachable without a
GLM model file.

## Re-enable

```
make clean
make DS4_ENABLE_GLM=1
```

This adds `-DDS4_ENABLE_GLM` to `CFLAGS`/`OBJCFLAGS` and restores the GLM
test targets (`make DS4_ENABLE_GLM=1 test-glm-attention`, etc.).
