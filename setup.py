import setuptools

setuptools.setup(
    name='Mynet',
    version='0.1.0',
    author='Andrey Volobuev',
    author_email='avvolob@gmail.com',
    packages=['mynet'],
    url='https://github.com/andreyvolobuev/mynet',
    license='MIT',
    description='Educational project to understand how neural networks actually work',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)