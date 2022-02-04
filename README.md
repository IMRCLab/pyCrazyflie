# pyCrazyflie
The `controller.py` is the main file of this repository. First of all, the dynamics of the UAV is encoded within `uavDy/uav.py`. Secondly, the trajectory optimization method using cvxpy is implemented in `genTrajectory/trajectoryGen.py`. 
___
## `genTrajectory/trajectoryGen.py`

* [ ] The *traj_choice* string variable let you choose the type of the trajectory
    * a linear trajectory (in the else part). Choose the x, y or \z trajectories by changing the index in *data* array.
    * infinity trajectories (change the string value of _type_traj_ = _inf_ )
    * Similarly for helix, *type_traj* = _helix_. 
    * Choose the pieces and the total time desired.
    * the waypoints are chosen equidistant with respect to the pieces and the middle conditions are all computed implicitly. 
* [ ] The output of this script generates a _.csv_ file saved in `/trajectoriescsv`. 
* [ ] In order to generate the trajectory, open the terminal in the `/genTrajectory` directory and run 
```bash
python3 trajectoryGen.py 
```
* [ ]  Then you will find the generated *.csv* file in `/trajectoriescsv` saved with the name of '*type_traj*'.csv variable.
___
## `controller.py`
* [ ] To run the simulator, open the terminal in `/pyCrazyflie` directory and type 
```bash
python3 controller.py 
``` 
* [ ] In order to choose whether to generate a video *.mp4* generated in the main directory, or show just a plot of the body frame, you will find a boolean *animateAndSave*. If true, the video will be generated otherwise a plot will be shown.
* [ ] For the frequency of the frames to be shown in the plotting (and similar for the video), change the *sample* variable. 
* [ ] Finally, the `initialize.py` lets you change the initial state of the UAV.