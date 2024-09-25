from setuptools import setup, find_packages
import subprocess

def check_gpu_availability():
    """Check if a GPU is available and return the appropriate ONNX Runtime package."""
    try:
        # Try to run a command that checks for GPU availability
        # The 'nvidia-smi' command is specific to NVIDIA GPUs.
        subprocess.check_output(['nvidia-smi'])
        return 'onnxruntime-gpu'
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If the command fails, assume no GPU is available
        return 'onnxruntime'

# Determine the correct ONNX Runtime package
onnx_runtime_package = check_gpu_availability()

setup(
    name='ton_ocr',
    version='0.2.3',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        onnx_runtime_package,  # Conditionally install the appropriate ONNX Runtime
        'shapely',
        'pyclipper'
    ],
    package_data={
        'ton_ocr': [
            'models/text_detection.onnx',
            'models/text_recognition.onnx',
            'rec/ppocr_keys_v1.txt',
        ],  # Include the ONNX files
    },
    description='A description of your project',
    author='Tung Nguyen',
    author_email='tungnguyen99.tn@gmail.com',
    url='https://github.com/tungedng2710/TonAI-OCR',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
