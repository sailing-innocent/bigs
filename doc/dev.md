# Development

## Intall Dependencies

### Init with uv (Recommmend)

- `uv sync`
- Install Local CUDA Extensions
  - `uv pip install .\lib\simple_knn\ --no-build-isolation`
  - `uv pip install .\lib\vanilla_3dgs\ --no-build-isolation`

### Check Install Valid

Run all the test suites to make sure all the dependencies are installed correctly:

`uv run pytest`

## Frequent Q & A

- make sure `CUDA_HOME` environment variable is set correctly
- make sure using developer powershell/cmd in Windows MSVC


