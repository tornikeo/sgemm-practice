# Single-precision general matrix multiplication (sgemm)

*Now with meson*. 

To build, make sure to `pip install` [meson](https://mesonbuild.com/) and [ninja](https://ninja-build.org/). And [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) or [conda](https://anaconda.org/anaconda/conda), `micromamba install conda-forge::cudatoolkit-dev`

> [!NOTE]
> You will need to `locate libcublas.so` and `export LD_LIBRARY_PATH=<...>`, where `<...>` is a dir which contains the `libcublas.so`. On my system this is `/home/tornikeo/micromamba/envs/pb/lib/`.

Compile:

```sh
meson setup builddir
meson compile -C builddir
```

Run:
```sh
# Test cuBLAS itself. Note % of cuBLAS could vary (at times over 100%)
meson test -C builddir/ sgemm_0 -vt 0

# Test naive kernel
meson test -C builddir/ sgemm_1 -vt 0

# Test progressively better kernels 2,3,...
meson test -C builddir/ sgemm_2 -vt 0
```