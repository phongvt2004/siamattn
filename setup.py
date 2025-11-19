from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os

# Get the directory containing this setup.py
base_dir = os.path.dirname(os.path.abspath(__file__))
toolkit_utils_dir = os.path.join(base_dir, 'toolkit', 'utils')
toolkit_utils_src_dir = os.path.join(toolkit_utils_dir, 'src')

ext_modules = [
    Extension(
        name='toolkit.utils.region',
        sources=[
            'toolkit/utils/region.pyx',
            'toolkit/utils/src/region.c',
        ],
        include_dirs=[
            toolkit_utils_src_dir,
            toolkit_utils_dir,  # Include directory containing c_region.pxd
        ],
        language='c'
    )
]

setup(
    name='toolkit',
    packages=['toolkit'],
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"})
)
