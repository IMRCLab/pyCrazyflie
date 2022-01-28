#!/usr/bin/env python

import cffirmware

def main():
    cffirmware.controllerSJCInit()

    # Allocate output variable
    # For this example, only thrustSI, and torque members are relevant
    control = cffirmware.control_t()

    # allocate desired state
    setpoint = cffirmware.setpoint_t()
    # select trajectory tracking mode
    setpoint.mode.x = cffirmware.modeAbs
    setpoint.mode.y = cffirmware.modeAbs
    setpoint.mode.z = cffirmware.modeAbs
    setpoint.mode.roll = cffirmware.modeDisable
    setpoint.mode.pitch = cffirmware.modeDisable
    setpoint.mode.yaw = cffirmware.modeDisable

    # allocate current state (both sensor and state are the state)
    # This is kind of odd and should be part of state
    sensors = cffirmware.sensorData_t()
    state = cffirmware.state_t()

    # Note that 1 tick == 1ms
    # note that the attitude controller will only compute a new output at 500 Hz
    # and the position controller only at 100 Hz
    # If you want an output always, simply select tick==0

    for tick in range(0, 100):

        # update desired state
        setpoint.position.x = tick*0.01     # m
        setpoint.position.y = 0             # m
        setpoint.position.z = 1.0           # m
        setpoint.velocity.x = 0             # m/s
        setpoint.velocity.y = 0             # m/s
        setpoint.velocity.z = 0             # m/s
        setpoint.acceleration.x = 0         # m/s^2
        setpoint.acceleration.y = 0         # m/s^2
        setpoint.acceleration.z = 0         # m/s^2
        setpoint.attitude.yaw = 0           # deg
        setpoint.attitudeRate.roll  = 0     # deg/s
        setpoint.attitudeRate.pitch = 0     # deg/s
        setpoint.attitudeRate.yaw   = 0     # deg/s

        # update current state
        sensors.gyro.x = 0 # deg/s
        sensors.gyro.y = 0 # deg/s # WARNING: THIS LIKELY NEEDS TO BE INVERTED
        sensors.gyro.z = 0 # deg/s

        state.attitudeQuaternion.w = 1
        state.attitudeQuaternion.x = 0
        state.attitudeQuaternion.y = 0
        state.attitudeQuaternion.z = 0
        state.position.x = 0    # m
        state.position.y = 0    # m
        state.position.z = 0    # m
        state.velocity.x = 0    # m/s
        state.velocity.y = 0    # m/s
        state.velocity.z = 0    # m/s
        state.acc.x      = 0    # Gs
        state.acc.y      = 0    # Gs
        state.acc.z      = 0    # Gs (without considering gravity)

        # query the controller
        cffirmware.controllerSJC(control, setpoint, sensors, state, tick)
        # result is in the following variables (units: N and Nm)
        print(control.thrustSI, control.torque)


if __name__ == '__main__':
	main()