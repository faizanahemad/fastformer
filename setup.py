from setuptools import setup, find_packages

setup(
  name = 'performer-pytorch',
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
  install_requires=[
    'einops>=0.3',
    'local-attention>=1.1.1',
    'pytorch-fast-transformers>=0.3.0',
    'torch>=1.7',
    'transformers>=3.5',
  ],
  classifiers=[
    'Development Status :: 1 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.9',
  ],
)
