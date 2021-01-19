from setuptools import setup, find_packages

setup(
  name = 'fastformer',
  packages = find_packages(exclude=['examples']),
  version = '0.0.1',
  license='MIT',
  description = 'Performer - Pytorch',
  author = 'Faizan Ahemad',
  author_email = 'fahemad3@gmail.com',
  url = 'https://github.com/faizanahemad/FAAST',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'efficient attention',
    'approximate attention',
    'transformers',
    'adversarial training',
  ],
  install_requires=[],
  classifiers=[
    'Development Status :: 1 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.9',
  ],
)
