from setuptools import setup, find_packages

setup(
    name="swan",
    version="0.01",
    author="Alison Peard",
    author_email="alison.peard@gmail.com",
    description="Python code for SWAN nearshore wave model.",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "shapely",
        "pandas",
        "geopandas",
        "matplotlib",
        "numba",
        "xarray",
        "cmocean"
    ]
)