from setuptools import find_packages
from setuptools import setup

with open("README.md", "r") as file:
    long_description = file.read()

setup(
    # name
    name='blip',
    # current version
    #   MAJOR VERSION:  01
    #   MINOR VERSION:  00
    #   Maintenance:    00
    version='01.00.00',
    # descriptions
    description='Blips and Low-energy Interaction Pointnet.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords='',
    # my info
    author='Nicholas Carrara',
    author_email='ncarrara.physics@gmail.com',
    # where to find the source
    url='https://github.com/Neutron-Calibration-in-DUNE/Blip',
    # requirements
    install_reqs=[],
    # packages
    # package_dir={'':'blip'},
    packages=find_packages(
        # 'blip',
        exclude=['tests'],
    ),
    include_package_data=True,
    package_data={'': ['*.yaml']},
    # classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    python_requires='>3.7',
    # possible entry point
    entry_points={
        'console_scripts': [
            'blip = blip.programs.run_blip:run',
        ],
    },
)
