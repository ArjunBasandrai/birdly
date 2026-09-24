from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tensorrt as trt
import torch


def _torch_dtype(data_type: trt.DataType) -> torch.dtype:
    if data_type == trt.float16:
        return torch.float16
    if data_type == trt.float32:
        return torch.float32
    raise TypeError(f"Unsupported TensorRT tensor type: {data_type}")


@dataclass(frozen=True)
class TensorRTModel:
    runtime: trt.Runtime
    engine: trt.ICudaEngine
    context: trt.IExecutionContext
    input_name: str
    output_name: str
    input_dtype: torch.dtype
    output_dtype: torch.dtype
    device: torch.device
    stream: torch.cuda.Stream
    minimum_batch_size: int
    maximum_batch_size: int
    engine_size_bytes: int

    def __call__(self, image_batch: torch.Tensor) -> torch.Tensor:
        if image_batch.device != self.device:
            raise ValueError(
                f"TensorRT requires a tensor on {self.device}, not {image_batch.device}"
            )
        batch_size = image_batch.shape[0]
        if not self.minimum_batch_size <= batch_size <= self.maximum_batch_size:
            raise ValueError(
                f"TensorRT batch size must be from {self.minimum_batch_size} "
                f"through {self.maximum_batch_size}"
            )

        current_stream = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current_stream)
        with torch.cuda.stream(self.stream):
            engine_input = image_batch.to(dtype=self.input_dtype).contiguous()
            if not self.context.set_input_shape(
                self.input_name, tuple(engine_input.shape)
            ):
                raise RuntimeError("TensorRT rejected the input shape")
            output_shape = tuple(self.context.get_tensor_shape(self.output_name))
            if any(dimension < 1 for dimension in output_shape):
                raise RuntimeError(f"TensorRT output shape is invalid: {output_shape}")
            logits = torch.empty(
                output_shape,
                dtype=self.output_dtype,
                device=self.device,
            )
            if not self.context.set_tensor_address(
                self.input_name, engine_input.data_ptr()
            ):
                raise RuntimeError("TensorRT rejected the input address")
            if not self.context.set_tensor_address(self.output_name, logits.data_ptr()):
                raise RuntimeError("TensorRT rejected the output address")
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT inference failed")
        current_stream.wait_stream(self.stream)
        return logits.float()


def load_tensorrt_model(
    engine_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> TensorRTModel:
    engine_path = Path(engine_path)
    if not engine_path.is_file():
        raise FileNotFoundError(f"TensorRT engine is missing: {engine_path}")
    selected_device = torch.device(device)
    if selected_device.type != "cuda":
        raise ValueError("TensorRT requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device_index = selected_device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    if device_index != 0:
        raise ValueError("Birdly TensorRT supports CUDA device 0 only")
    selected_device = torch.device("cuda", device_index)

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with torch.cuda.device(selected_device):
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("TensorRT could not deserialize the engine")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT could not create an execution context")

    input_names = []
    output_names = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        elif mode == trt.TensorIOMode.OUTPUT:
            output_names.append(name)
    if len(input_names) != 1 or len(output_names) != 1:
        raise ValueError("Birdly TensorRT must have one input and one output")

    input_name = input_names[0]
    output_name = output_names[0]
    minimum_shape, _, maximum_shape = engine.get_tensor_profile_shape(input_name, 0)
    return TensorRTModel(
        runtime=runtime,
        engine=engine,
        context=context,
        input_name=input_name,
        output_name=output_name,
        input_dtype=_torch_dtype(engine.get_tensor_dtype(input_name)),
        output_dtype=_torch_dtype(engine.get_tensor_dtype(output_name)),
        device=selected_device,
        stream=torch.cuda.Stream(device=selected_device),
        minimum_batch_size=int(minimum_shape[0]),
        maximum_batch_size=int(maximum_shape[0]),
        engine_size_bytes=engine_path.stat().st_size,
    )
