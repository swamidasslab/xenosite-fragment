from pathlib import Path
from setuptools import setup


# Loads _version.py module without importing the whole package.
def get_version_and_cmdclass(pkg_path):
    import os
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location(
        "version",
        os.path.join(pkg_path, "_version.py"),
    )
    module = module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(module)  # type: ignore
    return module.__version__, module.get_cmdclass(pkg_path)


version, cmdclass = get_version_and_cmdclass("xenosite/fragment")

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="xenosite-fragment",
    version=version,
    cmdclass=cmdclass,
    description="Library for molecule fragment operations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="S. Joshua Swamidass",
    author_email="swamidass@wustl.edu",
    packages=["xenosite/fragment"],
    entry_points={
        "console_scripts": [
            "xenosite-fragment=xenosite.fragment.__main__:app",
        ],
        "xenosite_command": ["fragment=xenosite.fragment.__main__:app"],
    },
    install_requires=["rdkit", "numpy", "numba", "rich", "typer", "networkx", "pandas"],
    classifiers=[
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Typing :: Typed",
    ],
)
