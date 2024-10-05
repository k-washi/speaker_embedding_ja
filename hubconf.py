"""torch.hub configuration."""

dependencies = ["torch"]

import torch
from pathlib import Path
import gdown


from models.ecapatdnn import _SpeakerEmbeddingJa

def ecapatdnn_ja_l512(progress: bool = True, pretrained: bool = True) -> _SpeakerEmbeddingJa:
    model = _SpeakerEmbeddingJa(hidden_size=512)
    if pretrained:
        output_fp = Path("/tmp/speaker-emb-ja-ecapa-tdnn-l512.pth")
        gdown.download(
            "https://drive.google.com/u/1/uc?id=16f191MtRAGOYB_LBZT8SMQGylqeW89U3",
            str(output_fp),
            quiet=False
        )
        model.model.load_state_dict(torch.load(output_fp, weights_only=True))
    
    model.model.eval()
    model.preprocess.eval()
    for param in model.model.parameters():
        param.requires_grad = False
    return model