from numpy.distutils.core import setup

setup(
    # Self-descriptive entries which should always be present
    name='dpyscf',
    author='Sebastian Dick',
    author_email='sebastian.dick@stonybrook.edu',
    license='BSD-3-Clause',

    # Which Python importable modules should be included when your package is installed
    packages=['dpyscf'],
    # Optional include package data to ship with your package
    # Comment out this line to prevent the files from being packaged with your software
    # Extend/modify the list to include/exclude other items as need be
    # Additional entries you may want simply uncomment the lines you want and fill in the data
    # author_email='me@place.org',      # Author email
    # url='http://www.my_package.com',  # Website
    # install_requires=[],              # Required packages, pulls from pip if needed; do not use for Conda deployment
    # platforms=['Linux',
    #            'Mac OS-X',
    #            'Unix',
    #            'Windows'],            # Valid platforms your code works on, adjust to your flavor
    # python_requires=">=3.5",          # Python version restrictions

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,
)
