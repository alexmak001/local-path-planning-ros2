from setuptools import setup
import os
from glob import glob

package_name = 'motion_plan_pkg'

submodule_name = 'graph_ltpl'
submodule_name2 = 'params'
submodule_name3 = 'inputs'
submodule = str(package_name + '/' + submodule_name)
submodule2 = str(package_name + '/' + submodule_name2)
submodule3 = str(package_name + '/' + submodule_name3)

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name,submodule,submodule2,submodule3],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, submodule_name), glob(submodule_name + '/*.py')),
        (os.path.join('share', package_name, submodule_name2), glob(submodule_name2 + '/*.ini')),
        (os.path.join('share', package_name, submodule_name3), glob(submodule_name3 + '/*.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motion_plan = motion_plan_pkg.motion_plan:main' #tweak
        ],
    },
)
