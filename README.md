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
./builddir/sgemm 0  # Run unoptimized kernel and comapre with cublas
```