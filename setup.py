"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from os import path
from setuptools import setup
# io.open is needed for projects that support Python 2.7
# It ensures open() defaults to text mode with universal newlines,
# and accepts an argument to specify the text encoding
# Python 3 only projects can skip this import
from io import open


# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.
def _parse_requirements(file_path):
    with open(file_path) as fp:
        reqs = [r.rstrip() for r in fp.readlines() if not r.startswith('#')]
        return reqs


HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as fp:
    long_description = fp.read()

install_reqs = _parse_requirements(path.join(HERE, 'requirements.txt'))

setup(
    name='yolo3',  # Required
    version='0.1',  # Required
    packages=['yolo3'],  # Required
    description='YOLO v3 in Keras',  # Required
    author='qqwweee',  # Optional
    author_email='qqwweee',  # Optional
    long_description=long_description,  # Optional (see note above)
    # long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/Borda/keras-yolo3',  # Optional
    install_requires=install_reqs,  # Optional
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha; 4 - Beta; 5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',
        # Specify the Python versions you support HERE. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # 'Programming Language :: Python :: 2',
        # 'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='Yolo CNN object-detector',  # Optional
)
