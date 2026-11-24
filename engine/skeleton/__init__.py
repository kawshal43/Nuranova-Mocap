"""Articulated skeletal solver package."""
from .rig_definition import (
    BONE_NAMES,
    BONE_INDEX,
    PARENT_INDICES,
    DEFAULT_REST_OFFSETS_M,
    ANATOMICAL_LIMITS_DEG,
    ActorProfile,
    compute_forward_kinematics_frame,
    compute_forward_kinematics_sequence,
    clamp_bone_rotation,
    calibrate_actor_from_body26_frames,
)
from .solver import (
    SolverConfig,
    ArticulatedSkeletonSolver,
    solve_two_bone_chain_rotations,
)

__all__ = [
    "BONE_NAMES",
    "BONE_INDEX",
    "PARENT_INDICES",
    "DEFAULT_REST_OFFSETS_M",
    "ANATOMICAL_LIMITS_DEG",
    "ActorProfile",
    "compute_forward_kinematics_frame",
    "compute_forward_kinematics_sequence",
    "clamp_bone_rotation",
    "calibrate_actor_from_body26_frames",
    "SolverConfig",
    "ArticulatedSkeletonSolver",
    "solve_two_bone_chain_rotations",
]
