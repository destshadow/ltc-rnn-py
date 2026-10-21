"""Collect standalone checks without requiring local trained checkpoints."""

import importlib

import pytest
import torch


@pytest.mark.parametrize("module_name", [
    "check_config", "check_cell", "check_classifier", "check_event_order",
    "check_temporal_order", "check_sequence_modes", "check_solver_validation",
    "check_solver_dynamics", "check_controller", "check_controller_stream",
    "check_controller_loading",
])
def test_portable_check(module_name):
    # Rewrite assertions in the imported standalone scripts too.
    pytest.register_assert_rewrite(module_name)
    importlib.import_module(module_name).main()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA non disponibile")
@pytest.mark.parametrize("module_name", [
    "check_environment", "check_state", "check_neuron_parameters",
    "check_synapse_parameters", "check_synapses", "check_solver",
])
def test_cuda_check(module_name):
    pytest.register_assert_rewrite(module_name)
    importlib.import_module(module_name).main()
