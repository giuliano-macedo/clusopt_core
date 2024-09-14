from pathlib import Path
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension
from functools import partial
from os import environ

# this is set automatically by the CI, change this to the new version you want if you plan to release do pypi locally.
version = "CHANGE ME"

# streamkm original source files are written in C++ syntax but with a .c extension
# so setting the compiler to g++ forces c++ compilation
if "CC" not in environ:
    environ["CC"] = "g++"

compiler_args = ["-O2", "-std=c++11", "-g0"]

# # for debug use this one
# compiler_args = ["-O2", "-std=c++11", "-D DEBUG"]

MyExtension = partial(
    Pybind11Extension,
    extra_compile_args=compiler_args,
    extra_link_args=compiler_args,
    language="c++",
)

clustream_path = Path("clusopt_core/cluster/clustream")
streamkm_path = Path("clusopt_core/cluster/streamkm")

path_glob = lambda path, pattern: sorted(map(str, path.rglob(pattern)))
ext_modules = [
    MyExtension(
        "clusopt_core.metrics.silhouette",
        ["clusopt_core/metrics/silhouette.cpp"],
    ),
    MyExtension(
        "clusopt_core.metrics.dist_matrix",
        ["clusopt_core/metrics/dist_matrix.cpp"],
        libraries=["boost_system", "boost_thread"],
    ),
    MyExtension(
        "clusopt_core.cluster.clustream.clustream",
        path_glob(clustream_path, "*.cpp"),
    ),
    MyExtension(
        "clusopt_core.cluster.streamkm.streamkm",
        path_glob(streamkm_path, "*.cpp") + path_glob(streamkm_path, "*.c"),
    ),
]

setup(
    name="clusopt_core",
    license="GPLv3",
    version=version,
    author="Giuliano Oliveira De Macedo",
    author_email="giuliano.llpinokio@gmail.com",
    description="Clustream, Streamkm++ and metrics utilities C/C++ bindings for python",
    long_description=Path("./README.md").read_text(),
    download_url=f"https://github.com/giuliano-oliveira/clusopt_core/archive/v{version}.tar.gz",
    long_description_content_type="text/markdown",
    keywords=["data-stream", "clustering", "silhouette"],
    url="https://github.com/giuliano-oliveira/clusopt_core",
    packages=find_packages(),
    install_requires=Path("requirements.txt").read_text().split(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    ext_modules=ext_modules,
)