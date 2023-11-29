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
    install_requires=[
        "auto-gptq==0.5.0.dev0",
        "triton==2.1.0",
        "fastapi==0.103.2",
        "uvicorn==0.23.2",
        "python-dotenv==1.0.0"
    ]
)
