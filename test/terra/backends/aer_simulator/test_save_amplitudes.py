# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Integration Tests for SaveAmplitudes instruction
"""

import cmath
import random

from ddt import ddt
import numpy as np
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT


@ddt
class TestSaveAmplitudes(SimulatorTestCase):
    """SaveAmplitudes instruction tests."""

    AMPLITUDES = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [7, 6, 5, 4, 3, 2, 1, 0],
        [5, 3, 0, 2],
        [0],
        [5, 2],
        [7, 0],
    ]

    def _normalize_amplitude_params(self, params):
        """Convert amplitude params to integer indices for target lookup."""
        normalized = []
        for param in params:
            if isinstance(param, str):
                if param.startswith("0x"):
                    normalized.append(int(param, 16))
                else:
                    normalized.append(int(param, 2))
            else:
                normalized.append(param)
        return normalized

    def _test_save_amplitudes(self, circuit, params, amp_squared, **options):
        """Test save_amplitudes instruction"""
        backend = self.backend(**options)

        # Stabilizer test circuit
        circ = circuit.copy()

        # Target statevector
        target = qi.Statevector(circ).data[self._normalize_amplitude_params(params)]
        if amp_squared:
            target = np.abs(target) ** 2

        # Add save to circuit
        label = "amps"
        if amp_squared:
            circ.save_amplitudes_squared(params, label=label)
        else:
            circ.save_amplitudes(params, label=label)

        # Run
        result = backend.run(transpile(circ, backend, optimization_level=0), shots=1).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        value = simdata[label]
        self.assertTrue(np.allclose(value, target))

    @supported_methods(
        ["automatic", "statevector", "matrix_product_state", "tensor_network"], AMPLITUDES
    )
    def test_save_amplitudes(self, method, device, params):
        """Test save_amplitudes instruction"""
        self._test_save_amplitudes(QFT(3), params, False, method=method, device=device)

    @supported_methods(
        [
            "automatic",
            "statevector",
            "matrix_product_state",
            "density_matrix",
            "tensor_network",
        ],
        AMPLITUDES,
    )
    def test_save_amplitudes_squared(self, method, device, params):
        """Test save_amplitudes_squared instruction"""
        self._test_save_amplitudes(QFT(3), params, True, method=method, device=device)

    @supported_methods(
        [
            "automatic",
            "stabilizer",
            "statevector",
            "matrix_product_state",
            "density_matrix",
            "tensor_network",
        ],
        AMPLITUDES,
    )
    def test_save_amplitudes_squared_clifford(self, method, device, params):
        """Test save_amplitudes_squared instruction for Clifford circuit"""
        # Stabilizer test circuit
        circ = QuantumCircuit(3)
        circ.h(0)
        circ.cx(0, 1)
        circ.x(2)
        circ.sdg(1)
        self._test_save_amplitudes(circ, params, True, method=method, device=device)

    @supported_methods(["statevector"], AMPLITUDES)
    def test_save_amplitudes_cache_blocking(self, method, device, params):
        """Test save_amplitudes instruction"""
        self._test_save_amplitudes(
            QFT(3),
            params,
            False,
            method=method,
            device=device,
            blocking_qubits=2,
            max_parallel_threads=1,
        )

    @supported_methods(["statevector", "density_matrix"], AMPLITUDES)
    def test_save_amplitudes_squared_cache_blocking(self, method, device, params):
        """Test save_amplitudes_squared instruction"""
        self._test_save_amplitudes(
            QFT(3),
            params,
            True,
            method=method,
            device=device,
            blocking_qubits=2,
            max_parallel_threads=1,
        )

    @supported_methods(["automatic", "statevector", "matrix_product_state", "tensor_network"])
    def test_save_amplitudes_hex_params(self, method, device):
        """Test save_amplitudes with hexadecimal basis-state parameters."""
        params = ["0x0", "0x3", "0x5", "0x7"]
        self._test_save_amplitudes(QFT(3), params, False, method=method, device=device)

    @supported_methods(["matrix_product_state"])
    def test_save_amplitudes_mps_bitstring_over_64_qubits(self, method, device):
        """Reproduce issue #2410 for MPS with >64 qubits bitstring params."""
        nqubits = 70
        qc = QuantumCircuit(nqubits)

        for q in range(nqubits):
            qc.p(2 * np.pi * (2**q) / (2**nqubits), q)

        x_int = random.Random(2410).randrange(2**nqubits)
        x_bits = format(x_int, f"0{nqubits}b")

        prep = QuantumCircuit(nqubits)
        for q, bit in enumerate(x_bits[::-1]):
            if bit == "1":
                prep.x(q)
        qc.compose(prep, front=True, inplace=True)

        qc.save_amplitudes([x_bits], label="amp")

        backend = self.backend(
            method=method,
            device=device,
            matrix_product_state_truncation_threshold=0.0,
            fusion_enable=False,
        )
        result = backend.run(qc, shots=1).result()

        self.assertTrue(result.success)
        amp = result.data(0)["amp"][0]
        target = cmath.exp(1.0j * 2 * np.pi * x_int / (2**nqubits))
        self.assertTrue(cmath.isclose(amp, target, rel_tol=1e-9, abs_tol=1e-9))
