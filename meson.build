project(
  'NVIDIA_SGEMM_PRACTICE',
  'cuda',
  version: '1.0',
  license: 'MIT',
  meson_version: '>=0.50.0',
)

srcs = []
subdir('src')

srcs += files(
  'sgemm.cu',
)

add_project_arguments('-lcudart -lcublas', language: 'cuda')
add_project_link_arguments('-lcudart -lcublas', language: 'cuda')

inc_dirs = include_directories('src')

executable('sgemm', srcs, include_directories: inc_dirs, install: false)