from setuptools import setup, find_packages

setup(
    name='photorch',
    version='1.4.3',
    author='Tong Lei, Kyle T. Rizzo, Brian N. Bailey',
    description='PhoTorch is Python-based software for robust fitting of photosynthesis and stomatal conductance models based on leaf-level gas exchange data. ',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'photorch': ['data/*.json'],  # adjust path/type as needed
    },
    install_requires=['torch','numpy','pandas','scipy','matplotlib'],  # Add dependencies here
    python_requires='>=3.6',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",  # ← tell PyPI it's Markdown
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
