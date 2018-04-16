from setuptools import setup, find_packages

setup(name='ml-engine-demo',
        version='0.1',
        packages=find_packages(),
        description='example to run tensorflow on gcloud ml-engine',
        author='Dan Mazur',
        author_email='danpmazur@gmail.com',
        install_requires=[
                  'pandas'
              ],
        zip_safe=False)
