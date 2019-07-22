# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='dbestclient',
    version='3.0',
    description='Model-based Approximate Query Processing (AQP) engine.',
    classifiers=[
        'Development Status :: 3.0',
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
    packages=['dbestclient'],#find_packages(exclude=('tests', 'docs','results'))
    zip_safe=False,
    install_requires=[
          'numpy'
      ],
    test_suite='nose.collector',
    tests_require=['nose'],
)
