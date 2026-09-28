"""The maneuvers that ship with nf_robot. AsyncObserver registers every one in BUILTIN_MANEUVERS."""

from nf_robot.host.maneuvers.parking import Parking

BUILTIN_MANEUVERS = (Parking,)
