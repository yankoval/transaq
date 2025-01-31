from setuptools import setup, find_packages

setup(
    name='market',
    version='1.0',
    packages=[],#find_packages(),#[''],
    py_modules=['moex','transaq','sentiment_indicator'],
    install_requires=['requests>=2.27.1',
    'pandas>=1.2.2',
    'numpy>=1.26.4',
    'mplfinance>=0.12',
    'coloredlogs>=15.0',
    'constants>=0.6.0',
    'tapy>=1.9.0',
    'scikit-learn>=1.5.1',
    'lxml>=5.3.0',
    'tqdm>=4.66.5'],
    url='',
    license='',
    author='ivan',
    author_email='',
    description='moex import utils'
)
