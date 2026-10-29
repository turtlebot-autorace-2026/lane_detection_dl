from setuptools import find_packages, setup
import glob

package_name = 'lane_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', glob.glob('models/*.pt') + glob.glob('models/*.onnx')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='asilia',
    maintainer_email='keummingi0131@gmail.com',
    description='Lane Detection package for turtlebot3_autorace_2025',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detection = lane_detection.detection:main',
        ],
    },
)
