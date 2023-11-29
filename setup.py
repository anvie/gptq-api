from setuptools import setup, find_packages

setup(
    name='gptq-api',
    version='0.0.3',
    author='Robin Syihab',
    license='MIT',
    author_email='robinsyihab@gmail.com',
    description='OpenAI compatible API server for AutoGPTQ model',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    include_package_data=True,
    packages=find_packages(),
    repository="https://github.com/anvie/gptq-api",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=2.7',
    install_requires=open('requirements.txt').read().split("\n")
)
