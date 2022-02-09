# pyCrazyflie
## Building Instructions
```
git clone --recursive git@github.com:IMRCLab/pyCrazyflie.git
cd crazyflie-firmware
make bindings_python
```
Add _cffirmware-firmware/_ to your `PYTHONPATH`. Open a terminal in the `pyCrazyflie/` directory and type
``` bash
 export PYTHONPATH=crazyflie-firmware/
```
## Simulation Structure
* [ ] The `controller.py` is the main file of this repository. The dynamics of the UAV is encoded within `uavDy/uav.py`.
* [ ] In order to execute a desired trajectory, add a `.csv` file in `trajectoriescsv/` folder in the following format:
  * [ ] each row must contain: time, desired position, desired velocity, desired acceleration. Such that, the desired position = [x, y, z], similarly for velocity and acceleration.
    ``` 
      *.csv file format
        rows: time, xd, yd, zd, vxd, vyd, vzd, axd, ayd, azd
    ```
### Main File: `controller.py`
* [ ] First of all, make sure that the filename variable in the _main_ function is the same as the `trajectoriescsv/*.csv` needed to executed. e.g. 

    ```python
    filename = "trajectoriescsv/inf.csv"
    ```
* [ ] To run the simulator, open the terminal in `pyCrazyflie/` directory and type 
    ```bash
    python3 controller.py 
    ``` 
* [ ] In order to choose whether to generate a video *.mp4* generated in the main directory, or show just a plot of the body frame, you will find a boolean *animateAndSave*. If true, the video will be generated otherwise a plot will be shown.
* [ ] For the frequency of the frames to be shown in the plotting (and similar for the video), change the *sample* variable. 
* [ ] Finally, the initial state of the UAV  and the time step _dt_ are intialized in the following function
```python
  def initializeState():
    """This function sets the initial states of the UAV
        dt: time step
        initPose: initial position [x,y,z]
        initq: initial rotations represented in quaternions 
        initLinVel: [xdot, ydot, zdot] initial linear velocities
        initAngVel: [wx, wy, wz] initial angular velocities"""
```
___
## Expected Output in Vidoes/ Directory
![Markdown Logo](Videos/infinitytraj.gif)
