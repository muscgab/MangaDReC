"""Public loader for MangaDReC."""
from __future__ import annotations
from pathlib import Path
import sys
from typing import Sequence
import torch

class MangaDReC:
    def __init__(self, model, characters: Sequence[str], spec: dict, device: torch.device):
        self.model = model.eval()
        self.characters = list(characters)
        self.spec = spec
        self.device = device

    @classmethod
    def from_pretrained(cls, checkpoint: str | Path | None = None, device=None):
        package = Path(__file__).resolve().parent
        runtime = package / "runtime"
        sys.path[:0] = [str(runtime / "tools"), str(runtime / "src"), str(runtime / "third_party" / "PaddleOCR2Pytorch")]
        import v1preview_all_gpu  # noqa: F401
        import fusion_b_student_model  # noqa: F401
        import fusion_b_top3_verifier  # noqa: F401
        import fusion_b_visual_top16_model  # noqa: F401
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                raise RuntimeError("MangaDReC requires MPS or CUDA/ROCm")
        target = torch.device(device)
        if target.type not in {"mps", "cuda"}:
            raise ValueError("device must be MPS or CUDA/ROCm")
        checkpoint = Path(checkpoint) if checkpoint is not None else package / "mangadrec_v1.pt"
        saved = torch.load(checkpoint, map_location=target, weights_only=False)
        return cls(saved["model"], saved["characters"], saved["spec"], target)

    @torch.inference_mode()
    def forward_gpu(self, images_bgr: torch.Tensor, source_sizes_hw: torch.Tensor):
        if images_bgr.device.type != self.device.type or source_sizes_hw.device.type != self.device.type:
            raise ValueError("both inputs must already be on the selected accelerator")
        return self.model(images_bgr, source_sizes_hw)

    def decode(self, output) -> list[str]:
        ids = output.token_ids.detach().cpu()
        counts = output.token_count.detach().cpu()
        return ["".join(self.characters[int(token) + 1] for token in row[:int(count)]) for row, count in zip(ids, counts, strict=True)]
