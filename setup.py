from setuptools import find_packages, setup

setup(
    name='domain_adaptation_se',
    version='1.0.0',
    description='Domain Adaptation for Speech Enhancement',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch==2.8.0',
        'torchaudio==2.8.0',
        'pytorch-lightning==2.2.5',
        'hydra-core==1.3.2',
        'omegaconf==2.3.0',
        'hydra-colorlog==1.2.0',
        'antlr4-python3-runtime==4.9.3',
        'librosa==0.10.2.post1',
        'soundfile==0.12.1',
        'scipy==1.13.1',
        'pystoi==0.4.1',
        'pesq==0.0.4',
        'h5py==3.11.0',
        'numpy==1.26.4',
        'matplotlib==3.9.0',
        'tqdm==4.67.1',
        'colorlog==6.9.0',
    ],
    extras_require={
        'data_gen': [
            'pyroomacoustics==0.7.4',
        ],
    },
)
