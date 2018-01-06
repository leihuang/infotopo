"""
"""

from setuptools import setup, find_packages



setup(name='infotopo',
      version='0.1.1',
      description=('A Python package for for information-geometric and '
                   'information-topological analysis of mathematical models'),
      packages=find_packages(exclude=['tests', 'docs']),
      url='https://github.com/leihuang/infotopo',
      license='MIT',
      author='Lei Huang',
      author_email='lh389@cornell.edu',
      install_requires=['pandas',
                        'numpy',
                        'matplotlib',
                        'scipy'],
      classifiers=['Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering :: Physics',
                   'Topic :: Scientific/Engineering :: Bio-Informatics'
                   ],
      )
