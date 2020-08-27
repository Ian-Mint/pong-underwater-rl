from setuptools import setup
from Cython.Build import cythonize

"""
run:

```
~/pong-underwater-rl$ python underwater_rl/setup.py build_ext --inplace
```
"""

setup(
    ext_modules=cythonize(
        "underwater_rl/replay.pyx",
        annotate=True,
        compiler_directives={'language_level': "3"}
    )
)
