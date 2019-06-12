from distutils.core import setup
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

setup(
  name='speechT',
  version='1.0',
  packages=['speecht'],
  scripts=['speecht-cli'],
  url='https://github.com/timediv/speechT',
  license='Apache License 2.0',
  author='Louis Kirsch',
  author_email='speechT@louiskirsch.com',
  description='An open source speech-to-text software written in tensorflow ',
  install_requires=[str(r.req) for r in parse_requirements("requirements.txt", session=False)]
)
