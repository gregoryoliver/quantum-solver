import heapq
import math
from typing import List, Optional, Tuple, Dict
import time
import gc
from itertools import product

import bittensor as bt
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps
import quimb.tensor as qtn
import cotengra as ctg
import numpy as np
import cupy as cp

LOG_FLOOR = 1e-300

class BeamTNOracle:
    def __init__(self, qc: QuantumCircuit, simplify_sequence: str = "ADCRS"):
        self.qc = qc
        self.n = qc.num_qubits  
        self.circ = qtn.Circuit(qc.num_qubits)
        qasm = dumps(qc)
        self.tqc = self.circ.from_openqasm2_str(qasm)
        print(f"Constructed tensor network for {self.n}-qubit circuit with {len(self.tqc.gates)} gates")
        self.simplify_sequence = simplify_sequence
        self.psi = self.tqc.get_psi_simplified(seq=simplify_sequence, atol=1e-10, equalize_norms=False)

        self.optimizer = ctg.ReusableHyperOptimizer(
            parallel=64,
            optlib="optuna",
            max_time="rate:1e8",
            max_repeats=32,
            directory=True,
            progbar=True,
            slicing_opts={'target_size': 2**29},
        )
        
        self.full_compress_optimizer = ctg.ReusableHyperCompressedOptimizer(
            chi=256,
            parallel=64,
            optlib="optuna",
            max_time="rate:1e8",
            max_repeats=400,
            directory=True,
            progbar=True,
            methods=['greedy-compressed', 'kahypar-agglom'],
        )
        
        self.compress_optimizer = ctg.ReusableHyperCompressedOptimizer(
            chi=256,
            parallel=64,
            optlib="optuna",
            max_time="rate:1e8",
            max_repeats=400,
            directory=True,
            progbar=True,
            methods=['greedy-compressed', 'kahypar-agglom'],
        )
            
        try:
            self._ensure_backend(self.psi, use_cupy=True)
        except Exception as e:
            print(f"Warning: failed to convert psi to CuPy: {e}")
    
    def compress_state_vector(self, prefix: Dict[int, int], use_cupy: bool = True) -> Tuple[float, str]:
        
        tn = self.psi.hyperinds_resolve(mode="tree")
        n = len(tn.outer_inds())
        remaining = [i for i in range(n-4) if i not in prefix]
        r = len(remaining)
        
        def ensure_cupy(tn):
            for t in tn.tensors:
                if isinstance(t.data, np.ndarray) and not isinstance(t.data, cp.ndarray):
                    t.modify(data=cp.asarray(t.data, dtype=cp.complex64))

        ensure_cupy(tn)

        tnc = tn.copy()
        out_inds = sorted(tnc.outer_inds(), key=lambda x: int(x[1:n]))
        
        for q, bit in prefix.items():
            idx = out_inds[q]
            vec = np.array([1, 0], dtype=np.complex64) if bit == 0 else np.array([0, 1], dtype=np.complex64)
            arr = cp.asarray(vec) if use_cupy else vec
            tnc |= qtn.Tensor(arr, inds=(out_inds[q],))
        
        for i in range(n-4, n):
            last_idx = out_inds[i]
            sum_vec = np.array([1 + 0j, 1 + 0j], dtype=np.complex64)
            sum_arr = cp.asarray(sum_vec) if use_cupy else sum_vec
            tnc |= qtn.Tensor(sum_arr, inds=(last_idx,))

        remaining_out_inds = [out_inds[q] for q in remaining]
        if use_cupy:
            backend_ctx = qtn.contract_backend('cupy')
        else:
            backend_ctx = qtn.null_context()
        with backend_ctx:
            amp_tensor = tnc.contract_compressed(optimize=self.compress_optimizer, output_inds=remaining_out_inds, max_bond=2**7)
        
        arr = cp.asarray(amp_tensor.data)
        p = float(cp.sum(cp.abs(arr)**2).item())
        
        state = arr.reshape(-1)
        
        probs = cp.abs(state)**2
        probs /= probs.sum()
        idx = int(cp.argmax(probs).item())
        size = int(cp.log2(len(state)))
        sub_bitstring = format(idx, f"0{r}b")
        
        del amp_tensor, arr, state, probs, tnc
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        
        return p, sub_bitstring
        
    def full_compress_state_vector(self, prefix: Dict[int, int], use_cupy: bool = True) -> Tuple[float, str]:
        
        tn = self.psi
        tn = tn.hyperinds_resolve(mode="tree")
        n = len(tn.outer_inds())
        remaining = [i for i in range(n) if i not in prefix]
        r = len(remaining)
        
        def ensure_cupy(tn):
            for t in tn.tensors:
                if isinstance(t.data, np.ndarray) and not isinstance(t.data, cp.ndarray):
                    t.modify(data=cp.asarray(t.data, dtype=cp.complex64))

        ensure_cupy(tn)

        tnc = tn.copy()
        out_inds = sorted(tnc.outer_inds(), key=lambda x: int(x[1:n]))
        
        for q, bit in prefix.items():
            idx = out_inds[q]
            vec = np.array([1, 0], dtype=np.complex64) if bit == 0 else np.array([0, 1], dtype=np.complex64)
            arr = cp.asarray(vec) if use_cupy else vec
            tnc |= qtn.Tensor(arr, inds=(out_inds[q],))

        remaining_out_inds = [out_inds[q] for q in remaining]
        if use_cupy:
            backend_ctx = qtn.contract_backend('cupy')
        else:
            backend_ctx = qtn.null_context()
        with backend_ctx:
            amp_tensor = tnc.contract_compressed(optimize=self.full_compress_optimizer, output_inds=remaining_out_inds, max_bond=2**7)
        
        arr = cp.asarray(amp_tensor.data)
        p = float(cp.sum(cp.abs(arr)**2).item())
        
        state = arr.reshape(-1)
        
        probs = cp.abs(state)**2
        probs /= probs.sum()
        idx = int(cp.argmax(probs).item())
        size = int(cp.log2(len(state)))
        sub_bitstring = format(idx, f"0{r}b")
        
        del amp_tensor, arr, state, probs, tnc
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        
        return p, sub_bitstring
        
    def verify_candidates_exact(self, candidates: list[str]) -> tuple[str, float]:
        best_p = -1.0
        best_str = ""
        print(f"candidate: {candidates}")
        for bitstr in candidates:
            try:
                prefix = {i: int(bit) for i, bit in enumerate(bitstr)}
                out_inds = sorted(self.psi.outer_inds(), key=lambda x: int(x[1:]))
                tnc = self.psi.copy()
                for q, bit in prefix.items():
                    vec = np.array([1, 0], dtype=np.complex64) if bit == 0 else np.array([0, 1], dtype=np.complex64)
                    arr = cp.asarray(vec)
                    tnc |= qtn.Tensor(arr, inds=(out_inds[q],))
                
                with qtn.contract_backend('cupy'):
                    amp = tnc.contract(optimize=self.optimizer, output_inds=[])
                
                if hasattr(amp, 'get'):
                    amp = cp.asnumpy(amp)
                p = float(abs(complex(amp)) ** 2)
                print(f"bitstr: {bitstr} -> {p}")
                if p > best_p:
                    best_p = p
                    best_str = bitstr
                
                # FEATURE: Explicit GPU memory freeing to prevent leaking
                del amp, tnc
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()
                    
            except Exception as e:
                print(f"Error verifying candidate {bitstr}: {e}")
                # FEATURE: Explicit GPU memory freeing to prevent leaking
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()
                continue

        return best_str, best_p

    def _ensure_backend(self, tn: "qtn.TensorNetwork", use_cupy: bool = True) -> None:
        try:
            if use_cupy:
                for tensor in tn.tensors:
                    data = tensor.data
                    if isinstance(data, np.ndarray) and not isinstance(data, cp.ndarray):
                        tensor.modify(data=cp.asarray(data))
            else:
                for tensor in tn.tensors:
                    data = tensor.data
                    if hasattr(data, "get") and isinstance(data, cp.ndarray):
                        tensor.modify(data=cp.asnumpy(data))
        except Exception as e:
            print(f"Backend normalization warning: {e}")
            
class CustomPeakSolver:

    def __init__(self):
        print("CustomPeakSolver initialized with GPU acceleration")

    def format_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.2f} secs"
        else:
            minutes = int(seconds // 60)
            rem_seconds = seconds % 60
            return f"{minutes} min {rem_seconds:.2f} secs"

    def beam_search(
        self,
        oracle: BeamTNOracle,
        cut_position: int = 5,
        ) -> List[Tuple[str, float]]:

        n = oracle.n
        print(f"Starting beam search with cut at position {cut_position} on {n}-qubit circuit")
        beam: List[Tuple[float, str]] = []
        t0 = time.perf_counter()
        k=0
        for prefix_bits in product('01', repeat=cut_position):
            prefix = ''.join(prefix_bits)
            prefix_dict = {i: int(b) for i, b in enumerate(prefix)}
            k += 1
            p = LOG_FLOOR
            sub_bitstring = ''
            try:
                if n < 36:
                    p, sub_bitstring = oracle.full_compress_state_vector(prefix_dict)
                else:
                    p, sub_bitstring = oracle.compress_state_vector(prefix_dict)
                    prefix = prefix + sub_bitstring[:6]
                    new_prefix_dict = {i: int(b) for i, b in enumerate(prefix)}
                    p, sub_bitstring = oracle.full_compress_state_vector(new_prefix_dict)
                p = max(p, LOG_FLOOR)
                
                complete_bitstring = prefix + sub_bitstring
                beam.append((math.log(p), complete_bitstring))
                print(f"{prefix} + {sub_bitstring} -> {math.log(p)} _{k}")
                         
            except Exception as e:
                beam.append((math.log(p), prefix))
                
        beam = heapq.nlargest(len(beam), beam, key=lambda x: x[0])
  
        total_t = time.perf_counter() - t0
        print(f"Completed in {self.format_time(total_t)}.")
        final_sorted = sorted(beam, key=lambda x: x[0], reverse=True)
        
        return [(bs, lp) for (lp, bs) in final_sorted]

    def solve(self, qasm: str) -> str:
        try:
            qc = QuantumCircuit.from_qasm_str(qasm)
            n = qc.num_qubits
            if n > 36:
                cut_position = n-32
            else:
                cut_position = 5
            try:
                overall_start = time.perf_counter()

                beam_tn_oracle = BeamTNOracle(qc)
                
                candidates = self.beam_search(beam_tn_oracle, cut_position=cut_position)
                
                if not candidates:
                    print("No candidates found, returning empty string")
                    return ""
                
                candidate_bslist = [bs for (bs, lp) in candidates]
                bitstring, p = beam_tn_oracle.verify_candidates_exact(candidate_bslist[:20])
                
                print(f"Best bitstring: {bitstring} with probability {p:.6e}")
                overall_end = time.perf_counter()
                print(f"Overall process took {self.format_time(overall_end - overall_start)}.")
                
                return bitstring
            except Exception as e:
                print(f"Failed to process QASM: {e}")
                return ""
        except Exception as e:
            print(f"Peaked solver failed: {e}")
            return ""