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
### Initialization
* [ ] The `config/initialize.yaml` sets the all the required initialization for each robot (e.g., path for the trajectory of each robot, initial conditions, dynamic parameters, etc...)
### Main File: `controller.py`
* [ ] To run the simulator, open the terminal in `pyCrazyflie/` directory and type the following command, providing an argument as the name of the pdf and video file that will be created after he simulation finishes running (i.e., choose a name for the file instead `name_of_the_file` )
  ```bash
    python3 controller.py name_of_the_file
    ```    
    
* [ ] In order to animate the simulation in a *.gif* format or save the generated plots in a pdf type this command
    ```bash
    python3 controller.py name_of_the_file --animate --plot
    ``` 
* [ ] `--animate` and `--plot` are flags. Their defaults are both False. 
* [ ] The animation will be saved in the `Videos` Directory, while the pdf will be saved in the main `pyCrazyflie` directory.
## Expected Output in Vidoes Directory
![Markdown Logo](Videos/infinity8.gif)

## TODOS:
* [ ] Add tests (with pytest) for different configuration of the config.yaml and examples folders.
* [ ] Use vispy.
* [ ] Add clipping for the rotors (to match the actual firmware).
* [ ] Add lee controller for multi UAVs case in python.