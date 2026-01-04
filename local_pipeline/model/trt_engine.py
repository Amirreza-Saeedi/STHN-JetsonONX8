from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import tensorrt as trt
try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Failed to import PyTorch. TensorRT inference in this repo uses PyTorch CUDA tensors as buffers (no pycuda). "
        "On Jetson this often means a missing CUDA runtime library such as libcufile.so.0. "
        "Try: sudo apt-get update && sudo apt-get install -y libcufile-12-6 && sudo ldconfig"
    ) from e


_TRT_TO_TORCH_DTYPE = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
}


@dataclass
class TrtIo:
    name: str
    is_input: bool
    dtype: torch.dtype


class TrtEngine:
    def __init__(self, engine_path: str, *, logger_severity: int = trt.Logger.ERROR):
        self.engine_path = engine_path
        self._logger = trt.Logger(logger_severity)
        self._runtime = trt.Runtime(self._logger)

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.engine = engine
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context: {engine_path}")

        self.io = self._introspect_io()

    def _introspect_io(self) -> Dict[str, TrtIo]:
        io: Dict[str, TrtIo] = {}

        # TensorRT 10 API: num_io_tensors / get_tensor_name / get_tensor_mode
        if hasattr(self.engine, "num_io_tensors"):
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                is_input = mode == trt.TensorIOMode.INPUT

                trt_dtype = self.engine.get_tensor_dtype(name)
                torch_dtype = _TRT_TO_TORCH_DTYPE.get(trt_dtype)
                if torch_dtype is None:
                    raise NotImplementedError(f"Unsupported TRT dtype {trt_dtype} for tensor '{name}'")

                io[name] = TrtIo(name=name, is_input=is_input, dtype=torch_dtype)
            return io

        # Fallback (older TRT): bindings
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            is_input = self.engine.binding_is_input(i)
            trt_dtype = self.engine.get_binding_dtype(i)
            torch_dtype = _TRT_TO_TORCH_DTYPE.get(trt_dtype)
            if torch_dtype is None:
                raise NotImplementedError(f"Unsupported TRT dtype {trt_dtype} for binding '{name}'")
            io[name] = TrtIo(name=name, is_input=is_input, dtype=torch_dtype)
        return io

    def infer(
        self,
        inputs: Dict[str, torch.Tensor],
        *,
        stream: Optional[torch.cuda.Stream] = None,
        allocate_outputs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if stream is None:
            stream = torch.cuda.current_stream()

        # Ensure inputs are contiguous CUDA tensors
        normalized_inputs: Dict[str, torch.Tensor] = {}
        for name, tensor in inputs.items():
            if name not in self.io:
                raise KeyError(f"Engine has no tensor named '{name}'. Available: {list(self.io.keys())}")
            if not self.io[name].is_input:
                raise ValueError(f"Tensor '{name}' is not an input")
            if not tensor.is_cuda:
                tensor = tensor.cuda(non_blocking=True)
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            normalized_inputs[name] = tensor

        # Set dynamic shapes if supported/needed
        if hasattr(self.context, "set_input_shape"):
            for name, tensor in normalized_inputs.items():
                try:
                    self.context.set_input_shape(name, tuple(tensor.shape))
                except Exception:
                    # Fixed-shape engines will throw if you try to set shape; ignore.
                    pass

        outputs: Dict[str, torch.Tensor] = {}
        if allocate_outputs:
            for name, meta in self.io.items():
                if meta.is_input:
                    continue

                if hasattr(self.context, "get_tensor_shape"):
                    shape = tuple(self.context.get_tensor_shape(name))
                else:
                    # Older TRT binding API
                    binding_idx = self.engine.get_binding_index(name)
                    shape = tuple(self.context.get_binding_shape(binding_idx))

                if any(d < 0 for d in shape):
                    raise RuntimeError(
                        f"Output shape for '{name}' is dynamic/unresolved: {shape}. "
                        "Provide inputs with concrete shapes and ensure the engine has a valid profile."
                    )

                outputs[name] = torch.empty(size=shape, device="cuda", dtype=meta.dtype)

        # Bind addresses
        if hasattr(self.context, "set_tensor_address"):
            for name, tensor in normalized_inputs.items():
                self.context.set_tensor_address(name, int(tensor.data_ptr()))
            for name, tensor in outputs.items():
                self.context.set_tensor_address(name, int(tensor.data_ptr()))

            ok = self.context.execute_async_v3(stream.cuda_stream)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 returned False")
        else:
            # Older TRT binding API path
            bindings = [0] * self.engine.num_bindings
            for name, tensor in normalized_inputs.items():
                idx = self.engine.get_binding_index(name)
                bindings[idx] = int(tensor.data_ptr())
            for name, tensor in outputs.items():
                idx = self.engine.get_binding_index(name)
                bindings[idx] = int(tensor.data_ptr())

            ok = self.context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v2 returned False")

        return outputs
