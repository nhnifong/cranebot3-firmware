"""The maneuvers that ship with nf_robot. AsyncObserver registers every one in BUILTIN_MANEUVERS."""

from nf_robot.host.maneuvers.cluster_sort import ClusterSort
from nf_robot.host.maneuvers.diagnostics import Diagnostics
from nf_robot.host.maneuvers.drop_point import DropPoint
from nf_robot.host.maneuvers.ferry import Ferry
from nf_robot.host.maneuvers.lerobot import Lerobot
from nf_robot.host.maneuvers.parking import Parking
from nf_robot.host.maneuvers.pick_and_place import PickAndPlace
from nf_robot.host.maneuvers.plates import Plates

BUILTIN_MANEUVERS = (Parking, DropPoint, Lerobot, PickAndPlace, Plates, Diagnostics, Ferry,
                     ClusterSort)
