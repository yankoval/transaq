from setuptools import setup, find_packages

setup(
    name='market',
    version='1.0',
    packages=[],#find_packages(),#[''],
    py_modules=['moex','transaq','sentiment_indicator'],
    install_requires=['requests>=2.27.1',
    'pandas',
    'numpy>=1.19.5',
    'mplfinance',
    'coloredlogs',
    'constants',
    'lxml',
    'tqdm'],
    url='',
    license='',
    author='ivan',
    author_email='yankoval@gmail.com',
    description='moex import utils'
)
