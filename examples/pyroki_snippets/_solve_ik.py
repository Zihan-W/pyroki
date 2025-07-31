"""
Solves the basic IK problem.
"""

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk


def solve_ik(
    robot: pk.Robot,
    target_link_name: str,
    target_wxyz: onp.ndarray,
    target_position: onp.ndarray,
    target_elbow_position: onp.ndarray,
    target_elbow_rot_quat: onp.ndarray,
    target_elbow_link_name:str,
    init_q: onp.ndarray,
) -> onp.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_name: String name of the link to be controlled.
        target_wxyz: onp.ndarray. Target orientation.
        target_position: onp.ndarray. Target position.

    Returns:
        cfg: onp.ndarray. Shape: (robot.joint.actuated_count,).
    """
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    target_link_index = robot.links.names.index(target_link_name)
    target_elbow_idx = robot.links.names.index(target_elbow_link_name)
    target_elbow_idx = jnp.array(target_elbow_idx)
    cfg = _solve_ik_jax(
        robot,
        jnp.array(target_link_index),
        jnp.array(target_wxyz),
        jnp.array(target_position),
        target_elbow_position_jax=jnp.array(target_elbow_position),
        target_elbow_rot_quat_jax=jnp.array(target_elbow_rot_quat),
        target_elbow_link_index=jnp.array(target_elbow_idx),
        init_q=jnp.array(init_q),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)
    return onp.array(cfg)


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    target_elbow_position_jax: jax.Array,
    target_elbow_rot_quat_jax: jax.Array,
    target_elbow_link_index: jax.Array,
    init_q: jax.Array,
) -> jax.Array:
    joint_var = robot.joint_var_cls(0)
    factors = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_position
            ),
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.limit_cost(
            robot,
            joint_var,
            weight=100.0,
        ),
        pk.costs.elbow_cost(
            robot,
            joint_var,
            target_elbow_position=target_elbow_position_jax,
            target_elbow_rot_quat=target_elbow_rot_quat_jax,
            target_elbow_link_index=target_elbow_link_index,
            pos_weight=10.0,
            ori_weight=2.0,
        ),
        pk.costs.smoothness_cost(
            joint_var,
            robot.joint_var_cls(init_q),
            weight=50.0
        )
    ]
    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[joint_var]
