# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='dbestclient',
    version='1.2',
    description='Model-based Approximate Query Processing (AQP) engine.',
    classifiers=[
        'Development Status :: 1.2',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Approximate Query Processing :: AQP :: Data Warehouse',
      ],
    keywords='Approximate Query Processing AQP',
    url='https://github.com/qingzma/DBEstClient',
    author='Qingzhi Ma',
    author_email='Q.Ma.2@warwick.ac.uk',
    long_description=readme,
    license=license,
    packages=['dbestclient'], #find_packages(exclude=('examples', 'docs')),
    entry_points = {
            'console_scripts': ['dbestclient=dbestclient.main:main','dbestcmd=dbestclient.main:cmd'],
        },
    zip_safe=False,
    install_requires=[
          'numpy','sqlparse','pandas','scikit-learn','qregpy', 'scipy', 'dill', 'matplotlib', 'torch'
      ],
    test_suite='nose.collector',
    tests_require=['nose'],
)
