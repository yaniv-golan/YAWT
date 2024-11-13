from cx_Freeze import setup, Executable
import sys
import os

# Get the directory containing your source code
src_dir = os.path.join(os.path.dirname(__file__), 'src')

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "packages": ["yawt", "transformers", "torch", "numpy", "tqdm", "requests", 
                "dotenv", "srt", "logging", "iso639", "stjlib"],
    "excludes": [],
    "include_files": [
        ("config/", "config/"),
        (src_dir, "src/")  # Include the entire src directory
    ],
    "path": [src_dir] + sys.path  # Add src directory to Python path
}

setup(
    name="YAWT",
    version="0.5.2",
    description="Yet Another Whisper-based Transcriber",
    options={"build_exe": build_exe_options},
    executables=[Executable("scripts/yawt_frozen.py", target_name="yawt")]
) 