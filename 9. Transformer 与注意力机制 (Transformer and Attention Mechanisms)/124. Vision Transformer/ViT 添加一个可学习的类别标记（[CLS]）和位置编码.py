class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # 可学习的类别标记
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer编码器
        self.transformer = TransformerEncoder(depth, embed_dim, num_heads, mlp_ratio)
        
        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        
        # 补丁嵌入 [B, n_patches, embed_dim]
        x = self.patch_embed(x)
        
        # 添加类别标记
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1+n_patches, embed_dim]
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # Transformer编码器
        x = self.transformer(x)
        
        # 取CLS标记的表示用于分类
        x = x[:, 0]  # [B, embed_dim]
        
        # 分类层
        x = self.norm(x)
        x = self.head(x)
        
        return x