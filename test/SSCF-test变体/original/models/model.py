# original/models/model.py
import torch
import torch.nn as nn
from types import SimpleNamespace

# --- CORRECTED Relative Import ---
from .ae import AE


class Model(nn.Module):
    def __init__(self, params: SimpleNamespace):
        super(Model, self).__init__()
        self.params = params
        self.ae = AE(params)
        num_classes = getattr(params, 'num_of_classes', 5)

        # [MODIFIED] 将输入维度从 512 修改为 256，以匹配 Encoder 的新输出
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, *args, **kwargs):
        # [NEW] 从 kwargs 抓取 domain_ids (即从 Dataset 中来的 subject_id)
        domain_ids = kwargs.get('domain_ids', None)
        
        recon, mu = self.ae(x, domain_ids=domain_ids)  # 传给 AE

        if mu.dim() == 3:
            pred = self.classifier(mu)  # Output (B, T, C)
        elif mu.dim() == 2:
            # This case might indicate an issue upstream or requires different handling
            print(f"[WARN] Original model expected mu with 3 dims, got {mu.shape}. Applying classifier.")
            pred = self.classifier(mu)  # Output (B, C)
        else:
            raise ValueError(f"Unexpected shape for mu: {mu.shape}")

        return pred, recon, mu, None, None

    def inference(self, x):
        self.eval()
        with torch.no_grad():
            mu = self.ae.encoder(x, domain_ids=None)
            if mu.dim() == 3:
                pred = self.classifier(mu)  # (B, T, C)
                # For inference, maybe return the mean prediction over time?
                # pred = pred.mean(dim=1) # -> (B, C)
                # Or keep as is, depending on evaluation needs
            elif mu.dim() == 2:
                pred = self.classifier(mu)  # (B, C)
            else:
                raise ValueError(f"Unexpected shape for mu in inference: {mu.shape}")
        return pred