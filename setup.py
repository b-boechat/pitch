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
    ),
    Extension(
        "sls_h_cy",
        ["sls_h_cy.pyx"]
    ),
    Extension(
        "sls_i_cy",
        ["sls_i_cy.pyx"]
    ),
    Extension(
        "baseline_sls_cy",
        ["baseline_sls_cy.pyx"]
    )
]

setup(
    name='cython_implementations',
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': '3'}, annotate=False),
    zip_safe=False
)