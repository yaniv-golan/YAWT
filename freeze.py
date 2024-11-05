import sys
import os
import torch
import glob
import transformers
import platform
from cx_Freeze import setup, Executable

# Get absolute paths
src_dir = os.path.abspath("src")
config_dir = os.path.abspath("config")

# Get PyTorch and Transformers library paths
torch_lib_dir = os.path.dirname(torch.__file__)
transformers_dir = os.path.dirname(transformers.__file__)

# Detect platform
PLATFORM = platform.system().lower()
ARCH = platform.machine().lower()

# Platform-specific configurations
platform_specific = {
    'darwin': {
        'bin_includes': ['libiomp5.dylib'],
        'lib_patterns': ['*.dylib'],
        'platform_name': 'macos'
    },
    'linux': {
        'bin_includes': ['libiomp5.so'],
        'lib_patterns': ['*.so'],
        'platform_name': 'linux'
    },
    'windows': {
        'bin_includes': ['libomp.dll'],
        'lib_patterns': ['*.dll'],
        'platform_name': 'win'
    }
}

# Get platform-specific settings
platform_config = platform_specific.get(PLATFORM, platform_specific['linux'])

# Collect library files
include_files = [
    (config_dir, "config"),
    (src_dir, "src"),
    (transformers_dir, "lib/transformers"),
]

# Add PyTorch dynamic libraries
for pattern in platform_config['lib_patterns']:
    for lib in glob.glob(os.path.join(torch_lib_dir, 'lib', pattern)):
        if os.path.isfile(lib):
            include_files.append((lib, os.path.join('lib', 'torch', 'lib', os.path.basename(lib))))

build_exe_options = {
    "packages": [
        "yawt", 
        "transformers", 
        "torch", 
        "numpy", 
        "tqdm", 
        "requests", 
        "dotenv", 
        "srt", 
        "logging", 
        "iso639", 
        "stjlib",
        "regex",
        "packaging",
        "filelock",
        "importlib_metadata",
        "huggingface_hub",
    ],
    "excludes": [],
    "include_files": include_files,
    "path": [src_dir] + sys.path,
    "include_msvcr": PLATFORM == 'windows',  # Only needed for Windows
    "bin_includes": platform_config['bin_includes'],
    "bin_path_includes": [os.path.join(torch_lib_dir, 'lib')],
    "zip_includes": [
        (os.path.join(transformers_dir, "utils"), "transformers/utils"),
        (os.path.join(transformers_dir, "models"), "transformers/models"),
    ]
}

# Create target name with platform and architecture
target_name = f"yawt-{platform_config['platform_name']}-{ARCH}"
if PLATFORM == 'windows':
    target_name += '.exe'

setup(
    name="YAWT",
    version="0.5.0",
    description="Yet Another Whisper-based Transcriber",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            script="scripts/yawt_frozen.py",
            target_name=target_name,
            base=None
        )
    ]
) 