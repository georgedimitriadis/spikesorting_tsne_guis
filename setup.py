from setuptools import setup

setup(name='spikesorting_tsne_guis',
      version='0.0.1',
      description='GUIs that allow cleaning of spike sorting (kilosort) results and manual curation based on the t-SNE embedding',
      url='https://github.com/georgedimitriadis/spikesorting_tsne_guis',
      author='George Dimitriadis',
      author_email='gdimitri@hotmail.com',
      license='MIT',
      packages=['spikesorting_tsne_guis'],
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    project_urls={
        'Documentation': 'https://georgedimitriadis.github.io/spikesorting_tsne_guis/',
        'Code repository': 'https://github.com/georgedimitriadis/spikesorting_tsne_guis',
    },
    include_package_data=True,
    package_data={
          # If any package contains *.ini files, include them
          '': ['*.ini'],
    },
    python_requires='>=3.5',
    zip_safe=False)


