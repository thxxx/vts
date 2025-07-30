from typing import TypeVar

import gradio as gr
import torch
from einops import einsum, rearrange
from torch import Tensor, nn
from tqdm import tqdm

from torchode import status_codes
from torchode.problems import InitialValueProblem
from torchode.single_step_methods.base import SingleStepMethod
from torchode.solution import Solution
from torchode.step_size_controllers.controller import AdaptiveStepSizeController
from torchode.terms import ODETerm

MethodState = TypeVar("MethodState")
InterpolationData = TypeVar("InterpolationData")
ControllerState = TypeVar("ControllerState")


class AutoDiffAdjoint(nn.Module):
    def __init__(
        self,
        step_method: SingleStepMethod[MethodState, InterpolationData],
        step_size_controller: AdaptiveStepSizeController[ControllerState],
    ):
        super().__init__()

        self.step_method = step_method
        self.step_size_controller = step_size_controller

    def solve(
        self,
        problem: InitialValueProblem,
        term: ODETerm,
        pbar: gr.Progress | tqdm | None = None,
    ) -> Solution:
        step_method, step_size_controller = self.step_method, self.step_size_controller
        device, batch_size, ndim = problem.device, problem.batch_size, problem.ndim
        t_start, t_end, t_eval = problem.t_start, problem.t_end, problem.t_eval
        time_direction = problem.time_direction.to(dtype=t_start.dtype)

        ###############################
        # Initialize the solver state #
        ###############################

        t = t_start
        y = problem.y0
        stats_n_steps = y.new_zeros(batch_size, dtype=torch.long)
        stats_n_accepted = y.new_zeros(batch_size, dtype=torch.long)
        stats: dict[str, Tensor] = {}

        # Compute the boundaries in time to ensure that we never step outside of them
        t_min = torch.minimum(t_start, t_end)
        t_max = torch.maximum(t_end, t_start)

        y_eval = y.new_empty((problem.n_evaluation_points, *y.shape))

        # Keep track of which evaluation points have not yet been handled
        not_yet_evaluated = torch.ones_like(t_eval, dtype=torch.bool)

        # Normalize the time direction of the evaluation and end times for faster
        # comparisons
        minus_t_end_normalized = -time_direction * t_end
        minus_t_eval_normalized = -einsum(time_direction, t_eval, "b, b t -> b t")

        # Keep track of which solves are still running
        running = y.new_ones(batch_size, dtype=torch.bool)

        # Initialize additional statistics to track for the integration term
        stats["n_f_evals"] = torch.zeros(
            problem.batch_size, device="cpu", dtype=torch.long
        )

        # Compute an initial step size
        convergence_order = step_method.convergence_order()
        dt, controller_state, f0 = step_size_controller.init(
            term, problem, convergence_order, stats=stats
        )
        method_state = step_method.init(term, problem, f0, stats=stats)

        # Ensure that the initial dt does not step outside of the time domain
        dt = torch.clamp(dt, t_min - t, t_max - t)

        ##############################################
        # Take care of evaluation exactly at t_start #
        ##############################################

        # We copy the initial state into the evaluation if the first evaluation point
        # happens to be exactly `t_start`. This is required so that we can later assume
        # that rejection of the step (and therefore also no change in `t`) means that we
        # also did not pass any evaluation points.
        y_eval[0] = y
        not_yet_evaluated[:, 0] = False

        ####################################
        # Solve the initial value problems #
        ####################################

        # Iterate the single step method until all ODEs have been solved up to their end
        # point or any of them failed
        pbar_is_tqdm = pbar is None
        if pbar_is_tqdm:
            total = (
                None if self.step_size_controller.adaptive else problem.t_eval.shape[1]
            )
            pbar = tqdm(
                total=total, desc="Solving ODE...", dynamic_ncols=True, leave=True
            )
        while True:
            step_result, interp_data, method_state_next = step_method.step(
                term, y, t, dt, method_state, stats=stats
            )
            accept, dt_next, controller_state_next, status = (
                step_size_controller.adapt_step_size(
                    t, dt, y, step_result, controller_state, stats
                )
            )

            # Update the solver state where the step was accepted
            to_update = accept & running
            t = torch.where(to_update, t + dt, t)
            y = torch.where(
                to_update.reshape(-1, *([1] * (ndim - 1))), step_result.y, y
            )
            method_state = step_method.merge_states(
                to_update, method_state_next, method_state
            )

            #####################
            # Update statistics #
            #####################

            stats_n_steps.add_(running)
            stats_n_accepted.add_(to_update)

            ##################################
            # Update solver state and status #
            ##################################

            # Stop a solve if `t` has passed its endpoint in the direction of time
            running = torch.addcmul(minus_t_end_normalized, time_direction, t) < 0.0

            # We evaluate the termination condition here already and initiate a
            # non-blocking transfer to the CPU to increase the chance that we won't have
            # to wait for the result when we actually check the termination condition
            continue_iterating = torch.any(running) & torch.all(
                status == status_codes.SUCCESS
            )
            continue_iterating = continue_iterating.to("cpu", non_blocking=True)

            # There is a bug as of pytorch 1.12.1 where non-blocking transfer from
            # device to host can sometimes gives the wrong result, so we place this
            # event after the transfer to ensure that the transfer has actually happened
            # by the time we evaluate the result.
            if device.type == "cuda":
                continue_iterating_done = torch.cuda.Event()
                continue_iterating_done.record(torch.cuda.current_stream(device))
            else:
                continue_iterating_done = None

            #########################
            # Evaluate the solution #
            #########################

            # Evaluate the solution at all evaluation points that have been passed in
            # this step.
            #
            # This causes a blocking CPU-GPU sync point at to_be_evaluated.any
            # when t_eval is not None, but deferring this sync doesn't seem to
            # yield a speedup. A sync is necessary at each evaluation point anyway
            # since nonzero produces a variable-shape result.
            # See https://github.com/martenlienen/torchode/issues/46
            to_be_evaluated = (
                torch.addcmul(
                    minus_t_eval_normalized,
                    rearrange(time_direction, "b -> b ()"),
                    rearrange(t, "b -> b ()"),
                )
                >= 0.0
            ) & not_yet_evaluated
            if to_be_evaluated.any():
                interpolation = step_method.build_interpolation(interp_data)
                nonzero = to_be_evaluated.nonzero()
                sample_idx, eval_t_idx = nonzero[:, 0], nonzero[:, 1]
                y_eval[eval_t_idx, sample_idx] = interpolation.evaluate(
                    t_eval[sample_idx, eval_t_idx], sample_idx
                )

                not_yet_evaluated = torch.logical_xor(
                    to_be_evaluated, not_yet_evaluated
                )

            ########################
            # Update the step size #
            ########################

            # We update the step size and controller state only for solves which will
            # still be running in the next iteration. Otherwise, a finished instance
            # with an adaptive step size controller could reach infinite step size if
            # its final error was small and another instance is running for many steps.
            # This would then cancel the solve even though the "problematic" instance is
            # not even supposed to be running anymore.

            dt = torch.where(running, dt_next, dt)

            # Ensure that we do not step outside of the time domain, even for instances
            # that are not running anymore
            dt = torch.clamp(dt, t_min - t, t_max - t)

            controller_state = step_size_controller.merge_states(
                running, controller_state_next, controller_state
            )

            if continue_iterating_done is not None:
                continue_iterating_done.synchronize()
            pbar.update(1)
            if continue_iterating:
                continue

            ##################################################
            # Finalize the solver and construct the solution #
            ##################################################

            # Put the step statistics into the stats dict in the end, so that
            # we don't have to type-assert all the time in torchscript
            stats["n_steps"] = stats_n_steps
            stats["n_accepted"] = stats_n_accepted

            # The finalization scope is in the scope of the while loop so that the
            # `t_eval is None` case can access the `interp_data` in TorchScript.
            # Declaring `interp_data` outside of the loop does not work because its type
            # depends on the step method.

            # Report the number of evaluation steps that have been initialized with
            # actual data at termination. Depending on the termination condition,
            # the data might be NaN or inf but it will not be uninitialized memory.
            #
            # As of torch 1.12.1, searchsorted is not implemented for bool tensors,
            # so we convert to int first.
            stats["n_initialized"] = torch.searchsorted(
                not_yet_evaluated.int(),
                torch.ones((batch_size, 1), dtype=torch.int, device=device),
            ).squeeze(dim=1)

            if pbar_is_tqdm:
                pbar.close()
            return Solution(ts=t_eval, ys=y_eval, stats=stats, status=status)

    def __repr__(self):
        return (
            f"AutoDiffAdjoint(step_method={self.step_method}, "
            f"step_size_controller={self.step_size_controller}, "
            f"max_steps={self.max_steps}"
        )
