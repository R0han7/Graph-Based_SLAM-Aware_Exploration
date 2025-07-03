from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'cpp_solver'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),  # ✅ Automatically finds 'cpp_solver' folder
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'param'), glob('param/*')),
        (os.path.join('share', package_name, 'world'), glob('world/**/*', recursive=True)),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'msg'), glob('msg/*')),
        (os.path.join('share', package_name, 'srv'), glob('srv/*')),
        (os.path.join('share', package_name), ['exploration.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Alberto Soragna',
    maintainer_email='alberto.soragna@gmail.com',
    description='Graph-Based SLAM-Aware Exploration package',
    license='MIT',
    entry_points={
        'console_scripts': [
            'path_planner = cpp_solver.path_planner:main',
            'exploration_button_server = cpp_solver.exploration_button_server:main',
            'exploration_button_client = cpp_solver.exploration_button_client:main',
            'my_planner = cpp_solver.my_planner:main',
            'update_distance = cpp_solver.update_distance:main',
        ],
    },
)

