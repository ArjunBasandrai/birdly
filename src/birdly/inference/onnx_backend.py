from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

try:
    import torch
except ModuleNotFoundError:
    torch = None


@dataclass(frozen=True)
class OnnxRuntimeModel:
    session: ort.InferenceSession
    input_name: str
    output_name: str
    device: Any
    output_width: int
    execution_provider: str

    def __call__(self, image_batch: Any) -> Any:
        if isinstance(image_batch, np.ndarray):
            if str(self.device) != "cpu":
                raise ValueError("NumPy input requires ONNX Runtime CPU inference")
            if image_batch.dtype != np.float32:
                raise ValueError("Birdly ONNX inference requires an FP32 array")
            return self.session.run(
                [self.output_name],
                {self.input_name: np.ascontiguousarray(image_batch)},
            )[0]

        if torch is None:
            raise TypeError("ONNX Runtime input must be a NumPy array")
        if image_batch.device != self.device:
            raise ValueError(
                f"ONNX Runtime requires a tensor on {self.device}, "
                f"not {image_batch.device}"
            )
        if image_batch.dtype != torch.float32:
            raise ValueError("Birdly ONNX inference requires an FP32 tensor")
        if self.device.type == "cuda":
            image_batch = image_batch.contiguous()
            logits = torch.empty(
                (image_batch.shape[0], self.output_width),
                dtype=torch.float32,
                device=self.device,
            )
            binding = self.session.io_binding()
            binding.bind_input(
                name=self.input_name,
                device_type="cuda",
                device_id=self.device.index or 0,
                element_type=np.float32,
                shape=tuple(image_batch.shape),
                buffer_ptr=image_batch.data_ptr(),
            )
            binding.bind_output(
                name=self.output_name,
                device_type="cuda",
                device_id=self.device.index or 0,
                element_type=np.float32,
                shape=tuple(logits.shape),
                buffer_ptr=logits.data_ptr(),
            )
            self.session.run_with_iobinding(binding)
            return logits

        input_array = image_batch.detach().contiguous().numpy()
        logits = self.session.run(
            [self.output_name],
            {self.input_name: input_array},
        )[0]
        return torch.from_numpy(logits)


def load_onnx_runtime_model(
    model_path: str | Path,
    *,
    cpu_threads: int,
    device: Any = "cpu",
) -> OnnxRuntimeModel:
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model is missing: {model_path}")
    if torch is None:
        if str(device) != "cpu":
            raise RuntimeError("PyTorch is required for ONNX Runtime CUDA inference")
        selected_device: Any = "cpu"
    else:
        selected_device = torch.device(device)

    session_options = ort.SessionOptions()
    device_type = getattr(selected_device, "type", str(selected_device))
    if device_type == "cpu":
        session_options.intra_op_num_threads = cpu_threads
        providers: list[str | tuple[str, dict[str, object]]] = ["CPUExecutionProvider"]
        execution_provider = "CPUExecutionProvider"
    elif device_type == "cuda":
        if torch is None:
            raise RuntimeError("PyTorch is required for ONNX Runtime CUDA inference")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("ONNX Runtime CUDAExecutionProvider is unavailable")
        device_index = selected_device.index or 0
        if device_index != 0:
            raise ValueError("Birdly ONNX Runtime supports CUDA device 0 only")
        selected_device = torch.device("cuda", device_index)
        with torch.cuda.device(selected_device):
            stream = torch.cuda.current_stream(selected_device)
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "user_compute_stream": str(stream.cuda_stream),
                    "do_copy_in_default_stream": "1",
                    "use_tf32": "0",
                },
            ),
            "CPUExecutionProvider",
        ]
        execution_provider = "CUDAExecutionProvider"
    else:
        raise ValueError(f"Unsupported ONNX Runtime device: {selected_device}")

    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=providers,
    )
    if session.get_providers()[0] != execution_provider:
        raise RuntimeError(
            f"ONNX Runtime did not activate {execution_provider}: "
            f"{session.get_providers()}"
        )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Birdly ONNX must have one input and one output")
    if inputs[0].type != "tensor(float)" or outputs[0].type != "tensor(float)":
        raise ValueError("Birdly ONNX input and output must use FP32")
    output_width = outputs[0].shape[-1]
    if not isinstance(output_width, int):
        raise TypeError("Birdly ONNX output class dimension must be fixed")
    return OnnxRuntimeModel(
        session=session,
        input_name=inputs[0].name,
        output_name=outputs[0].name,
        device=selected_device,
        output_width=output_width,
        execution_provider=execution_provider,
    )
