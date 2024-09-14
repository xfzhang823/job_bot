import setuptools

setuptools.setup(
    name="your_project_name",
    version="0.1.0",
    author="Xiao-Fei Zhang",
    author_email="nextexit101@email.com",
    description="A bot to optimize resumes.",
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
