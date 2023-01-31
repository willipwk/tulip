## Requirements 
Python3.8
(Alternative branch on Python3.10 is inactively WIP)

## Dependencies
* Install lightweight motion planning package mplib
  * For Ubuntu ```pip install mplib```
  * For MacOS, manually install dependent package first with 
    ```
    brew install eigen ompl fcl pinocchnio orocos-kdl 
    git clone --recursive https://github.com/haosulab/MPlib.git
    git clone -r git+https://github.com/haosulab/MPlib.git
    ```


##  Installation
```
git clone git@github.com:friolero/tulip.git
cd tulip
pip install ".[dev]"
```
