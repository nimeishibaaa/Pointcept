import torch
import torch.nn as nn
from pointcept.models.builder import MODELS
from pointcept.models.losses import build_criteria

@MODELS.register_module("OpenVocabPPT")
class OpenVocabPPT(nn.Module):
    """
    OpenVocabPPT: An open-vocabulary 3D representation encoder.
    Takes dynamic text descriptions, encodes them via CLIP, and computes
    cosine similarity with 3D point tokens. Optimized with Binary Focal/BCE loss
    for open-vocabulary retrieval tasks.
    """
    def __init__(
        self,
        backbone=None,
        criteria=None,
        backbone_out_channels=96,
        template="[x]",
        clip_model="ViT-B/16",
        freeze_backbone=False,
    ):
        super().__init__()
        self.backbone = MODELS.build(backbone)
        self.criteria = build_criteria(criteria)
        self.template = template
        self.freeze_backbone = freeze_backbone
        
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
                
        import clip
        clip_model, _ = clip.load(clip_model, device="cpu", download_root="./.cache/clip")
        clip_model.requires_grad_(False)
        self.clip_model = clip_model
        
        self.proj_head = nn.Linear(backbone_out_channels, clip_model.text_projection.shape[1])
        self.logit_scale = clip_model.logit_scale

    def forward(self, data_dict):
        if self.freeze_backbone:
            with torch.no_grad():
                point = self.backbone(data_dict)
        else:
            point = self.backbone(data_dict)
            
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
            
        feat = self.proj_head(point.feat)
        eps = 1e-6 if feat.dtype == torch.float16 else 1e-12
        feat = nn.functional.normalize(feat, dim=-1, p=2, eps=eps)
        
        # In open-vocabulary mode, we expect texts to be provided dynamically in data_dict.
        # texts should be a list of strings for the current scene, e.g. ["background", "red spatula", "hammer"]
        assert "texts" in data_dict, "OpenVocabPPT requires 'texts' list in data_dict. The dataset must provide dynamic text queries."
        texts = data_dict["texts"]
        
        device = feat.device
        prompts = [self.template.replace("[x]", t) for t in texts]
        
        import clip
        text_tokens = clip.tokenize(prompts).to(device)
        
        # Ensure clip_model is on the correct device
        if next(self.clip_model.parameters()).device != device:
            self.clip_model = self.clip_model.to(device)
            
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        # sim shape: [N_points, Num_texts]
        sim = feat @ text_features.t()
        logit_scale = self.logit_scale.exp()
        seg_logits = logit_scale * sim
        
        if self.training:
            # target segment mask shape: [N_points, Num_texts]
            target_mask = data_dict["segment"].float()
            loss = self.criteria(seg_logits, target_mask)
            return dict(loss=loss)
        elif "segment" in data_dict.keys():
            target_mask = data_dict["segment"].float()
            loss = self.criteria(seg_logits, target_mask)
            return dict(loss=loss, seg_logits=seg_logits)
        else:
            return dict(seg_logits=seg_logits)
