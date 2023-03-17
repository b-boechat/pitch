from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "fls",
        ["fls.pyx"]
    ),
    Extension(
        "swgm",
        ["swgm.pyx"]
    )
]

setup(
    name='cython_implementations',
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': '3'}, annotate=False),
    zip_safe=False
)