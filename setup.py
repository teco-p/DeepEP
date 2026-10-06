"""Build the SDAA backend using the pinned native CMake toolchain."""
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

class NativeBuild(build_ext):
    def build_extension(self, ext):
        root = Path(__file__).resolve().parent
        target = Path(self.build_temp).resolve() / "sdaa"
        target.mkdir(parents=True, exist_ok=True)
        import torch
        import torch_sdaa
        sdk = Path(os.environ["SDAA_HOME"])
        backend = Path(torch_sdaa.__file__).resolve().parent
        python_include = Path(sysconfig.get_path("include"))
        if not all((python_include / name).is_file() for name in ("Python.h", "pyconfig.h")):
            raise RuntimeError(f"Current Python installation has no development header: {python_include}")
        subprocess.check_call(["cmake", "-S", str(root / "csrc/sdaa"), "-B", str(target),
            "-DBUILD_TESTING=OFF", "-DCMAKE_BUILD_TYPE=Release",
            "-DSDAA_INCLUDE_DIR=" + str(sdk / "include"),
            "-DSDAA_COMPILER=" + str(sdk / "bin/tecocc"),
            "-DTORCH_SDAA_INCLUDE_DIR=" + str(backend / "include"),
            "-DPython3_EXECUTABLE=" + sys.executable,
            "-DPython3_INCLUDE_DIR=" + str(python_include),
            "-DCMAKE_PREFIX_PATH=" + torch.utils.cmake_prefix_path,
            "-DCMAKE_INSTALL_PREFIX=" + str(Path(self.build_lib).resolve() / "deep_ep")])
        subprocess.check_call(["cmake", "--build", str(target), "--target", "_sdaa", "-j", os.environ.get("MAX_JOBS", "4")])
        subprocess.check_call(["cmake", "--install", str(target)])

setup(name="deep_ep", version="2.5.0+sdaa", packages=find_packages(),
      ext_modules=[Extension("deep_ep._sdaa", sources=[])], cmdclass={"build_ext": NativeBuild})
