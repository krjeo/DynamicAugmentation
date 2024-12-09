import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dynamicaug",
    version="0.0.1",
    author="JeonghyunKim",
    author_email="kr.jeonghyun.kim@gmail.com",
    description="schooldevops sample lib",
    long_description="will be added",
    long_description_content_type="text/markdown",
    url="https://github.com/krjeo/DynamicAugmentation",
    project_urls={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
)