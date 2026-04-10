# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
AerSimulator Integration Tests
"""
from copy import deepcopy
from concurrent.futures import Executor, ThreadPoolExecutor
from ddt import ddt
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates import HGate, XGate, ZGate
from qiskit.quantum_info import Statevector

from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods
from qiskit_aer.backends.aer_compiler import BACKEND_RUN_ARG_TYPES, _validate_option
from qiskit_aer.backends import StatevectorSimulator
from qiskit_aer.aererror import AerError
from qiskit_aer.noise import NoiseModel


@ddt
class TestVariousCircuit(SimulatorTestCase):
    """AerSimulator tests to simulate various types of circuits"""

    _VALID_TYPE_SAMPLES = {
        int: 1,
        np.integer: np.int64(1),
        float: 0.5,
        np.floating: np.float64(0.5),
        bool: True,
        np.bool_: np.bool_(True),
        str: "ok",
        list: [],
        NoiseModel: NoiseModel,
    }

    _INVALID_TYPE_SAMPLES = {
        int: 1.5,
        np.integer: 1.5,
        float: "1.5",
        np.floating: "1.5",
        bool: "true",
        np.bool_: "true",
        str: 123,
        list: "not-a-list",
        NoiseModel: {},
    }

    @staticmethod
    def _declared_types(expected_type):
        return expected_type if isinstance(expected_type, tuple) else (expected_type,)

    @classmethod
    def _valid_sample_for_type(cls, declared_type):
        if declared_type is Executor:
            return ThreadPoolExecutor(max_workers=1)
        sample = cls._VALID_TYPE_SAMPLES[declared_type]
        return sample() if callable(sample) else sample

    @classmethod
    def _invalid_sample_for_type(cls, declared_type):
        if declared_type is Executor:
            return 123
        return cls._INVALID_TYPE_SAMPLES[declared_type]

    @supported_methods(
        [
            "automatic",
            "statevector",
            "density_matrix",
            "matrix_product_state",
            "extended_stabilizer",
            "tensor_network",
        ]
    )
    def test_quantum_register_circuit(self, method, device):
        """Test circuits with quantum registers."""

        qubits = QuantumRegister(3)
        clbits = ClassicalRegister(3)

        circuit = QuantumCircuit(qubits, clbits)
        circuit.h(qubits[0])
        circuit.cx(qubits[0], qubits[1])
        circuit.cx(qubits[0], qubits[2])

        for q, c in zip(qubits, clbits):
            circuit.measure(q, c)

        backend = self.backend(method=method, device=device, seed_simulator=1111)

        shots = 1000
        result = backend.run(circuit, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, [circuit], [{"0x0": 500, "0x7": 500}], delta=0.05 * shots)

    @supported_methods(
        [
            "automatic",
            "statevector",
            "density_matrix",
            "matrix_product_state",
            "extended_stabilizer",
            "tensor_network",
        ]
    )
    def test_qubits_circuit(self, method, device):
        """Test circuits with quantum registers."""

        qubits = QuantumRegister(3)
        clbits = ClassicalRegister(3)

        circuit = QuantumCircuit()
        circuit.add_bits(qubits)
        circuit.add_bits(clbits)
        circuit.h(qubits[0])
        circuit.cx(qubits[0], qubits[1])
        circuit.cx(qubits[0], qubits[2])

        for q, c in zip(qubits, clbits):
            circuit.measure(q, c)

        backend = self.backend(method=method, device=device, seed_simulator=1111)

        shots = 1000
        result = backend.run(circuit, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, [circuit], [{"0x0": 500, "0x7": 500}], delta=0.05 * shots)

    @supported_methods(
        [
            "automatic",
            "statevector",
            "density_matrix",
            "matrix_product_state",
            "extended_stabilizer",
            "tensor_network",
        ]
    )
    def test_qubits_quantum_register_circuit(self, method, device):
        """Test circuits with quantum registers."""

        qubits0 = QuantumRegister(2)
        clbits1 = ClassicalRegister(2)
        qubits1 = QuantumRegister(1)
        clbits2 = ClassicalRegister(1)

        circuit = QuantumCircuit(qubits0, clbits1)
        circuit.add_bits(qubits1)
        circuit.add_bits(clbits2)
        circuit.h(qubits0[0])
        circuit.cx(qubits0[0], qubits0[1])
        circuit.cx(qubits0[0], qubits1[0])

        for qubits, clbits in zip([qubits0, qubits1], [clbits1, clbits2]):
            for q, c in zip(qubits, clbits):
                circuit.measure(q, c)

        backend = self.backend(method=method, device=device, seed_simulator=1111)

        shots = 1000
        result = backend.run(circuit, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, [circuit], [{"0x0": 500, "0x7": 500}], delta=0.05 * shots)

        qubits0 = QuantumRegister(1)
        clbits1 = ClassicalRegister(1)
        qubits1 = QuantumRegister(1)
        clbits2 = ClassicalRegister(1)
        qubits2 = QuantumRegister(1)
        clbits3 = ClassicalRegister(1)

        circuit = QuantumCircuit(qubits0, clbits1)
        circuit.add_bits(qubits1)
        circuit.add_bits(clbits2)
        circuit.add_register(qubits2)
        circuit.add_register(clbits3)
        circuit.h(qubits0[0])
        circuit.cx(qubits0[0], qubits1[0])
        circuit.cx(qubits1[0], qubits2[0])

        for qubits, clbits in zip([qubits0, qubits1, qubits2], [clbits1, clbits2, clbits3]):
            for q, c in zip(qubits, clbits):
                circuit.measure(q, c)

        backend = self.backend(method=method, device=device, seed_simulator=1111)

        shots = 1000
        result = backend.run(circuit, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, [circuit], [{"0x0": 500, "0x7": 500}], delta=0.05 * shots)

    def test_partial_result_a_single_invalid_circuit(self):
        """Test a partial result is returned with a job with a valid and invalid circuit."""

        circuits = []
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        qc_2 = QuantumCircuit(50)
        qc_2.h(range(50))
        qc_2.measure_all()
        circuits.append(qc_2)
        circuits.append(qc)
        backend = self.backend()
        shots = 100
        result = backend.run(circuits, shots=shots, method="statevector").result()
        self.assertEqual(result.status, "PARTIAL COMPLETED")
        self.assertTrue(hasattr(result.results[1].data, "counts"))
        self.assertFalse(hasattr(result.results[0].data, "counts"))

    def test_metadata_protected(self):
        """Test metadata is consitently viewed from users"""

        qc = QuantumCircuit(2)
        qc.metadata = {"foo": "bar", "object": object}

        circuits = [qc.copy() for _ in range(5)]

        backend = self.backend()
        job = backend.run(circuits)

        for circuit in circuits:
            self.assertTrue("foo" in circuit.metadata)
            self.assertEqual(circuit.metadata["foo"], "bar")
            self.assertEqual(circuit.metadata["object"], object)

        deepcopy(job.result())

    def test_validate_option_registry_types_have_test_samples(self):
        """Guard test: any new declared type must have explicit test data."""
        declared_types = set()
        for expected in BACKEND_RUN_ARG_TYPES.values():
            declared_types.update(self._declared_types(expected))

        known_valid_types = set(self._VALID_TYPE_SAMPLES)
        known_valid_types.add(Executor)
        known_invalid_types = set(self._INVALID_TYPE_SAMPLES)
        known_invalid_types.add(Executor)

        missing_valid = declared_types - known_valid_types
        missing_invalid = declared_types - known_invalid_types
        self.assertFalse(
            missing_valid,
            f"Missing valid samples for types: {[t.__name__ for t in sorted(missing_valid, key=str)]}",
        )
        self.assertFalse(
            missing_invalid,
            f"Missing invalid samples for types: {[t.__name__ for t in sorted(missing_invalid, key=str)]}",
        )

    def test_validate_option_accepts_all_declared_types(self):
        """Every declared type for each option should be accepted."""
        executors = []
        try:
            for key, expected in BACKEND_RUN_ARG_TYPES.items():
                for declared_type in self._declared_types(expected):
                    value = self._valid_sample_for_type(declared_type)
                    if isinstance(value, ThreadPoolExecutor):
                        executors.append(value)
                    with self.subTest(
                        option=key,
                        declared_type=declared_type.__name__,
                        value_type=type(value).__name__,
                    ):
                        result = _validate_option(key, value)
                        self.assertIsNotNone(result)
                        self.assertIsInstance(result, expected)
        finally:
            for executor in executors:
                executor.shutdown(wait=False)

    def test_validate_option_rejects_incompatible_types(self):
        """Every declared type for each option should reject incompatible values."""
        for key, expected in BACKEND_RUN_ARG_TYPES.items():
            for declared_type in self._declared_types(expected):
                wrong_value = self._invalid_sample_for_type(declared_type)
                with self.subTest(
                    option=key,
                    declared_type=declared_type.__name__,
                    wrong_value=wrong_value,
                    wrong_type=type(wrong_value).__name__,
                ):
                    with self.assertRaises(TypeError):
                        _validate_option(key, wrong_value)

        int_option_keys = [
            key
            for key, expected in BACKEND_RUN_ARG_TYPES.items()
            if int in self._declared_types(expected)
        ]
        for key in int_option_keys:
            with self.subTest(option=key, wrong_value=True, reason="bool should not be accepted"):
                with self.assertRaises(TypeError):
                    _validate_option(key, True)

    def test_validate_option_accepts_numpy_scalar_alternatives(self):
        """Numpy scalar variants declared in registry should be accepted."""
        for key, expected in BACKEND_RUN_ARG_TYPES.items():
            if not isinstance(expected, tuple):
                continue

            numpy_value = None
            if np.integer in expected:
                numpy_value = np.int64(1)
            elif np.floating in expected:
                numpy_value = np.float64(0.5)
            elif np.bool_ in expected:
                numpy_value = np.bool_(True)

            self.assertIsNotNone(
                numpy_value,
                f"Tuple option '{key}' has no numpy scalar branch covered by this test.",
            )

            with self.subTest(option=key, value_type=type(numpy_value).__name__):
                result = _validate_option(key, numpy_value)
                self.assertIsNotNone(result)
                self.assertIsInstance(result, expected)

    def test_validate_option_bool_options_reject_integers(self):
        """Bool-typed options should not accept integer values."""
        bool_option_keys = [
            key
            for key, expected in BACKEND_RUN_ARG_TYPES.items()
            if bool in (expected if isinstance(expected, tuple) else (expected,))
        ]
        for key in bool_option_keys:
            with self.subTest(option=key, wrong_value=1):
                with self.assertRaises(TypeError):
                    _validate_option(key, 1)

    def test_validate_option_none_passthrough(self):
        """Option value None should pass through unchanged."""
        self.assertIsNone(_validate_option("shots", None))

    def test_validate_option_unknown_key_raises_aererror(self):
        """Unknown option name should raise AerError."""
        with self.assertRaises(AerError):
            _validate_option("__unknown_option__", 1)

    def test_numpy_integer_shots(self):
        """Test implicit cast of shot option from np.int_ to int."""

        backend = self.backend()

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        shots = 333

        for np_type in {
            np.int_,
            np.uint,
            np.short,
            np.ushort,
            np.intc,
            np.uintc,
            np.longlong,
            np.ulonglong,
        }:
            result = backend.run(qc, shots=np_type(shots), method="statevector").result()
            self.assertSuccess(result)
            self.assertEqual(sum([result.get_counts()[key] for key in result.get_counts()]), shots)

    def test_invalid_parameters(self):
        """Test gates with invalid parameter length."""

        backend = self.backend()

        class Custom(Gate):
            def __init__(self, label=None):
                super().__init__("p", 1, [], label=label)

            def _define(self):
                q = QuantumRegister(1, "q")
                qc = QuantumCircuit(q, name=self.name)
                qc._append(HGate(), [q[0]], [])
                self.definition = qc

        qc = QuantumCircuit(1)
        qc.append(Custom(), [0])
        qc.measure_all()

        try:
            backend.run(qc).result()
            self.fail("do not reach here")
        except Exception as e:
            self.assertTrue('"params" is incorrect length' in repr(e))

    def test_controlled_gates(self):
        """Test gates with control qubits"""
        backend = StatevectorSimulator()
        num_qubits = 4
        circuit = QuantumCircuit(num_qubits)
        cccx = XGate().control(num_ctrl_qubits=3, label=None, ctrl_state="100")
        circuit.x(2)
        circuit.compose(cccx, range(num_qubits), inplace=True)
        job = backend.run(circuit)
        state = job.result().get_statevector()
        ref_state = Statevector(circuit)
        self.assertEqual(state, ref_state)

        num_qubits = 3
        circuit = QuantumCircuit(num_qubits)
        cccz = ZGate().control(num_ctrl_qubits=2, label=None, ctrl_state="10")
        circuit.x(1)
        circuit.x(2)
        circuit.compose(cccz, range(num_qubits), inplace=True)
        job = backend.run(circuit)
        state = job.result().get_statevector()
        ref_state = Statevector(circuit)
        self.assertEqual(state, ref_state)
