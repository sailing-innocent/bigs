# Development

## Install Dependencies

### Init with uv (Recommmend)

- `uv sync`
- Install Local CUDA Extensions
  - `uv pip install .\lib\simple_knn\ --no-build-isolation`
  - `uv pip install .\lib\vanilla_3dgs\ --no-build-isolation`
  - !Important `uv pip install .\lib\bigs\ --no-build-isolation` (Our Core Algorithm Implementation lives here!)
  - (Optional) install sam2 `uv pip install <path_to_sam2>` and download pretrained model according to 

### Check Install Valid

Run all the test suites to make sure all the dependencies are installed correctly:

`uv run pytest`

## Use Toolkits

We also provide some useful toolkits for debugging named `sail`, which provides some features like **Visualizing Point Cloud**, **Visualizing Proxy Mesh** and **Navigation in Scene**

## Frequent Q & A

- make sure `CUDA_HOME` environment variable is set correctly
- make sure using developer powershell/cmd in Windows MSVC


