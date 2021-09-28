from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='tgchatbot',
    version='0.0.1',
    description='Telegram AI chatbot',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/osmr/tgchatbot',
    author='Oleg Sémery',
    author_email='osemery@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Communications :: Chat',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='telegram chatbot asr tts nlu',
    packages=find_packages(exclude=['others', '*.others', 'others.*', '*.others.*']),
    include_package_data=True,
    install_requires=['transformers', 'SentencePiece', 'aiogram'],
)
