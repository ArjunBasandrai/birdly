from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import onnxruntime as ort
import torch


@dataclass(frozen=True)
class OnnxRuntimeModel:
    session: ort.InferenceSession
    input_name: str
    output_name: str

    def __call__(self, image_batch: torch.Tensor) -> torch.Tensor:
        if image_batch.device.type != "cpu":
            raise ValueError("ONNX Runtime inference requires a CPU tensor")
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
) -> OnnxRuntimeModel:
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model is missing: {model_path}")

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = cpu_threads
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Birdly ONNX must have one input and one output")
    if inputs[0].type != "tensor(float)" or outputs[0].type != "tensor(float)":
        raise ValueError("Birdly ONNX input and output must use FP32")
    return OnnxRuntimeModel(
        session=session,
        input_name=inputs[0].name,
        output_name=outputs[0].name,
    )
