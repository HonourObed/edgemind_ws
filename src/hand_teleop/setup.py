from setuptools import find_packages, setup

package_name = 'hand_teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Samuel Abolarinwa Ayomide',
    maintainer_email='developer@example.com',
    description='MediaPipe to ROS 2 Teleoperation Package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arm_controller = hand_teleop.arm_controller:main',
            'gripper_bridge = hand_teleop.gripper_bridge:main',
            'unified_vision = hand_teleop.unified_vision_node:main',
        ],
    },
)
