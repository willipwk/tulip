"""A collection of utility functions for PyBullet simulation."""
import time

import pybullet as p

PBLT_TIMESTEP = 1 / 240.0


def init_pblt_sim(**kwargs) -> int:
    """Initialise a PyBullet simulation client.

    Args:
        kwargs: Simulation setting such as connection mode in kwargs.
    Returns:
        physics client id"""
    mode = kwargs.get("mode", "GUI").upper()
    assert mode in [
        "GUI",
        "DIRECT",
        "SHARED_MEMORY",
        "UDP",
        "TCP",
    ], "Invalid input simulation mode!"

    if mode == "GUI":
        sim_cid = p.connect(p.GUI)
    elif mode == "DIRECT":
        sim_cid = p.connect(p.DIRECT)
    elif mode == "SHARED_MEMORY":
        # todo(zyuwei) remove default port num
        port_num = kwargs.get("port_num", 1234)
        sim_cid = p.connect(p.SHARED_MOMORY, port_num)
    else:
        host_ip = kwargs.get("host_ip", "localhost")
        port_num = kwargs.get("port_num", 1234)
        if mode == "UDP":
            sim_cid = p.connect(p.UDP, host_ip, port_num)
        else:
            sim_cid = p.connect(p.TCP, host_ip, port_num)
    assert sim_cid >= 0, "Failed to create a pybullet client"

    timestep = kwargs.get("timestep", PBLT_TIMESTEP)
    p.setTimeStep(timestep)

    gravity_coef = kwargs.get("gravity_coef", -9.8)
    p.setGravity(0, 0, gravity_coef)

    return sim_cid


def step_sim(
    num_steps: int = 1, mode: str = "GUI", timestep: float = PBLT_TIMESTEP
) -> None:
    """Running step simulation.

    Args:
        num_steps: number of steps to simulate
        mode: simulation connection mode
        timestep: simulation step time
    """
    for _ in range(num_steps):
        p.stepSimulation()
        if mode.upper == "GUI":
            time.sleep(PBLT_TIMESTEP)
    return


def enable_torque_sensor(
    robot_id: int,
    joint_indices: list[int],
    sim_cid: int,
) -> None:
    """Enable force/torque sensor on target joint(s).

    Args:
        robot_id: robot PyBullet uniqueBodyId.
        joint_indices: target joint indexes to enable the sensor.
        sim_cid: PyBullet physicsClientId"""
    for joint_idx in joint_indices:
        p.enableJointForceTorqueSensor(
            bodyUniqueId=robot_id,
            jointIndex=joint_idx,
            enableSensor=True,
            physicsClientId=sim_cid,
        )
    return


def disable_torque_sensor(
    robot_id: int,
    joint_indices: list[int],
    sim_cid: int,
) -> None:
    """Disable force/torque sensor on target joint(s).

    Args:
        robot_id: robot PyBullet uniqueBodyId.
        joint_indices: target joint indexes to disable the sensor.
        sim_cid: PyBullet physicsClientId"""
    for joint_idx in joint_indices:
        p.enableJointForceTorqueSensor(
            bodyUniqueId=robot_id,
            jointIndex=joint_idx,
            enableSensor=False,
            physicsClientId=sim_cid,
        )
    return
