import setuptools

setuptools.setup(
    name="your_project_name",
    version="0.1.0",
    author="Your Name",
    author_email="your@email.com",
    description="A brief description of your project",
    long_description="A longer description of your project",
    long_description_content_type="text/markdown",
    url="https://github.com/xfzhang823/job_bot",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
