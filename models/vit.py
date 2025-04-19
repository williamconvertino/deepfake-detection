import torch
import torch.nn as nn
import timm

class ViT(nn.Module):
    def __init__(self, num_classes=2, num_prompts=5, freeze_backbone=True):
        super().__init__()
        
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)

        self.prompt_dim = self.vit.embed_dim
        self.num_prompts = num_prompts

        self.vit.head = nn.Identity()

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        self.prompt_embeddings = nn.Parameter(torch.randn(1, num_prompts, self.prompt_dim))

        self.head = nn.Linear(self.prompt_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.size()

        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        pos_embed = self.vit.pos_embed
        prompt = self.prompt_embeddings.expand(B, -1, -1)
        
        # [CLS] + [PROMPTS] + [PATCHES]
        x = torch.cat((cls_token, prompt, x), dim=1)

        # Resize positional embeddings
        total_len = x.shape[1]
        if pos_embed.shape[1] != total_len:
            pe_cls = pos_embed[:, :1]
            pe_patch = pos_embed[:, 1:]
            pe_patch = torch.nn.functional.interpolate(
                pe_patch.reshape(1, int(pe_patch.shape[1]**0.5), int(pe_patch.shape[1]**0.5), -1).permute(0, 3, 1, 2),
                size=(int((total_len - 1 - self.num_prompts)**0.5),) * 2,
                mode='bilinear',
                align_corners=False,
            ).permute(0, 2, 3, 1).reshape(1, -1, self.prompt_dim)
            pos_embed = torch.cat([pe_cls, torch.zeros_like(prompt), pe_patch], dim=1)

        x = x + pos_embed[:, :x.shape[1], :]

        x = self.vit.pos_drop(x)
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)

        # Only use CLS token
        cls_output = x[:, 0]
        return self.head(cls_output)
