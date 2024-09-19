from setuptools import setup, find_packages

setup(
    name='ton_ocr',               
    version='0.2.1',                 
    packages=find_packages(),        
    install_requires=[ 
        'numpy',         
        'opencv-python',
        'onnxruntime',
        'shapely',
        'pyclipper'
    ],
    package_data={
        'ton_ocr': ['models/text_detection.onnx',
                    'models/text_recognition.onnx',
                    'rec/ppocr_keys_v1.txt'],  # Include the ONNX file
    },
    description='A description of your project',   # Short description
    author='Tung Nguyen',              # Your name (or organization)
    author_email='tungnguyen99.tn@gmail.com',  # Your email
    url='https://github.com/tungedng2710/TonAI-OCR',  # URL of your project
    classifiers=[                    # Additional metadata (optional)
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
