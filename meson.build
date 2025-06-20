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

exe = executable('sgemm', srcs, include_directories: inc_dirs, install: false)

test('sgemm_0', exe, args: ['0'])
test('sgemm_1', exe, args: ['1'])
test('sgemm_2', exe, args: ['2'])
test('sgemm_3', exe, args: ['3'])
test('sgemm_4', exe, args: ['4'])
test('sgemm_5', exe, args: ['5'])
test('sgemm_6', exe, args: ['6'])
test('sgemm_7', exe, args: ['7'])
test('sgemm_8', exe, args: ['8'])