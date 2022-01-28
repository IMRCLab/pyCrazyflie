# pyCrazyflie
Examples for using the crazyflie-firmware Python bindings


## Building

```
git clone --recursive git@github.com:IMRCLab/pyCrazyflie.git
cd crazyflie-firmware
make bindings_python
```

Adjust your PYTHONPATH (See examples below), or copy cffirmware*.so file

### Controller

```
PYTHONPATH=crazyflie-firmware python3 controller.py
```