import os
import platform

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

import pybind11


def is_truthy_env(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


class BuildExt(build_ext):
    c_opts = {
        "msvc": [
            "/O2",
            "/std:c++20",
            "/DNDEBUG",
        ],
        "unix": [
            "-O3",
            "-DNDEBUG",
            "-funroll-loops",
        ],
    }

    l_opts = {
        "msvc": [],
        "unix": [],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        system = platform.system().lower()    # linux, darwin, windows
        machine = platform.machine().lower()  # x86_64, amd64, arm64, aarch64

        opts = self.c_opts.get(ct, []).copy()
        link_opts = self.l_opts.get(ct, []).copy()

        native = is_truthy_env("NATIVE")
        zen4 = is_truthy_env("ZEN4")
        avx512 = is_truthy_env("AVX512")
        aggressive = is_truthy_env("AGGRESSIVE")

        if ct == "unix":
            opts += [
                "-std=c++20",
                "-fPIC",
            ]

            # LTO is usually helpful on Linux, but often annoying on macOS.
            if system != "darwin":
                opts += ["-flto"]
                link_opts += ["-flto", "-lm"]

            if machine in ("x86_64", "amd64"):
                if native:
                    opts += [
                        "-march=native",
                        "-mtune=native",
                    ]
                    print("** Building ArborEnum with -march=native -mtune=native")

                elif zen4:
                    opts += [
                        "-march=znver4",
                        "-mtune=znver4",
                    ]
                    print("** Building ArborEnum for AMD Zen 4 / EPYC Genoa")

                elif avx512:
                    opts += [
                        "-mpopcnt",
                        "-mbmi",
                        "-mbmi2",
                        "-mlzcnt",
                        "-mavx512f",
                        "-mavx512dq",
                        "-mavx512bw",
                        "-mavx512vl",
                        "-mavx512vpopcntdq",
                    ]
                    print("** Building ArborEnum with explicit AVX-512 VPOPCNTDQ flags")

                elif aggressive:
                    opts += [
                        "-mpopcnt",
                        "-mbmi",
                        "-mbmi2",
                        "-mlzcnt",
                        "-mavx2",
                    ]
                    print("** Building ArborEnum with aggressive x86 AVX2/BMI flags")

                else:
                    opts += [
                        "-mpopcnt",
                    ]
                    print("** Building ArborEnum with x86 POPCNT support")

            elif machine in ("arm64", "aarch64"):
                # ARM64 has efficient bit operations through compiler codegen / NEON.
                if native:
                    opts += [
                        "-mcpu=native",
                    ]
                    print("** Building ArborEnum on ARM64 with -mcpu=native")
                elif aggressive:
                    print("** AGGRESSIVE requested on ARM64; no extra portable flags added")
                else:
                    print("** Building ArborEnum on ARM64; skipping x86-specific flags")

            else:
                print(
                    f"** Building ArborEnum on unknown Unix arch {machine}; "
                    "skipping architecture-specific flags"
                )

        elif ct == "msvc":
            # MSVC maps __popcnt64 through <intrin.h>
            if native or zen4 or avx512 or aggressive:
                print("** SIMD env flag requested with MSVC; using safe MSVC flags")
            else:
                print("** Building ArborEnum with MSVC safe optimized flags")

        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts

        print("** extra_compile_args:", " ".join(opts))
        print("** extra_link_args:", " ".join(link_opts))

        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        "arborenum._core",
        sources=[
            "src/arborenum/_core.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            "src/arborenum",
            "src/arborenum/cpp",
        ],
        language="c++",
    ),
]


setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)