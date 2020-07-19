# -*- coding: utf-8 -*-

import os

from setuptools import find_packages, setup

with open('README.rst') as f:
    readme = f.read()


# with open('LICENSE.txt') as f:
#     licenses = f.read()


setup(
    name='dbestclient',
    version='2.0',
    description='Model-based Approximate Query Processing (AQP) engine.',
    classifiers=[
        'Development Status :: 2.0',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Approximate Query Processing :: AQP :: Data Warehouse',
    ],
    keywords='Approximate Query Processing AQP',
    url='https://github.com/qingzma/DBEstClient',
    author='Qingzhi Ma',
    author_email='Q.Ma.2@warwick.ac.uk',
    long_description=readme,
    # license=licenses,
    # packages=['dbestclient'],
    packages=find_packages(exclude=('experiments', 'tests', 'docs')),
    entry_points={
        'console_scripts': ['dbestclient=dbestclient.main:main', 'dbestslave=dbestclient.main:slave', 'dbestmaster=dbestclient.main:master'],
    },
    zip_safe=False,
    install_requires=[
        'numpy', 'sqlparse', 'pandas', 'scikit-learn', 'qregpy', 'scipy', 'dill', 'matplotlib', 'torch', 'category_encoders', 'tox', 'sphinx', 'gensim',
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
)
