# NVIDIA SGEMM PRACTICE

*now with meson*


# Build


To build, make sure to get:

Via pip:
- [meson](https://mesonbuild.com/)
- [ninja](https://ninja-build.org/)

Via [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) or [conda](https://anaconda.org/anaconda/conda):
- cudatoolkit-dev

> [!NOTE]
> You will need to `locate libcublas.so` and `export LD_LIBRARY_PATH=<...>`, where `<...>` is a dir which contains the `libcublas.so`. On my system this is `/home/tornikeo/micromamba/envs/pb/lib/`.

Compile:

```sh
meson setup builddir
meson compile -C builddir
```

Run:
```sh
./builddir/sgemm 0  # Run unoptimized kernel to comapre with cublas
```