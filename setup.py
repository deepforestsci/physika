from setuptools import setup, find_packages

setup(name="physika",
    version="0.1.0",
    url="https://github.com/deepforestsci/physika",
    license='MIT',
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "ply",
        "numpy"
    ],
    packages=find_packages(),
    project_urls={
          'Source': 'https://github.com/deepforestsci/physika',
      },
    entry_points={
        "console_scripts": [
            "physika=physika.execute:main",
        ],
    },
    extras_require={
        "dev": ["pytest>=7"],
    },
)
