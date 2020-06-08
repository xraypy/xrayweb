#!/usr/bin/env python
from setuptools import setup
import os

version = '0.1'
with open(os.path.join('xrayweb', '_version.py')) as fh:
    for line in fh.readlines():
        line = line[:-1].strip()
        if line.startswith('__version'):
            words = line.split('=') + [' ', ' ']
            version = words[1].strip().replace("'", '').replace('"', '')


pkg_data = {'xrayweb.template': ['template/*'],
            'xrayweb.static': ['static/*']}

setup(name='xrayweb',
      version = version,
      author       = 'Alex Nicolellis and Matthew Newville',
      author_email = 'newville@cars.uchicago.edu',
      license      = 'MIT',
      description  = 'Web interface for X-ray Properties of the elements',
      install_requires=('sqlalchemy', 'numpy', 'scipy', 'xraydb', 'flask', 'plotly'),
      package_dir = {'xrayweb': 'xrayweb'},
      packages = ['xrayweb'],
      package_data = pkg_data,
      zip_safe=False,
)
