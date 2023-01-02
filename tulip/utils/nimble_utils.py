"""A collection of utility functions for nimble simulation."""

import logging

import nimblephysics as n


def init_sim(**kwargs) -> n.simulation.World:
    """Initialize nimble simulation

    Args:
        kwargs: Nimble simulation setting in flexible kwargs input.
    Returns:
        nimble simulation world."""
    world = n.simulation.World()
    return world


def init_vis(world: n.simulation.World, port_num: int = 8080) -> n.NimbleGUI:
    """Initialize web visualizer

    Args:
        world: simulation world.
        port_num: port number for the GUI server.
    Returns:
        the web GUI visualizer."""
    gui = n.NimbleGUI(world)
    gui.serve(port_num)
    return gui
