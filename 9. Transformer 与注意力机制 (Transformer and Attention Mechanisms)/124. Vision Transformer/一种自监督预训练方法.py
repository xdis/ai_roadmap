# MAE主要思想
# 1. 随机遮蔽大部分图像块(如75%)
# 2. 仅对可见块应用编码器
# 3. 添加遮蔽标记并用解码器重建原图

class MAE(nn.Module):
    def __init__(self, encoder, decoder_dim, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder  # ViT编码器
        self.mask_ratio = mask_ratio
        
        # 简化的解码器
        self.decoder = nn.TransformerDecoder(...)
        
    def forward(self, x):
        # 1. 创建遮蔽
        patches = self.patchify(x)
        mask = self.random_masking(patches.shape[0], self.mask_ratio)
        
        # 2. 编码可见块
        visible_patches = patches[~mask]
        encoded = self.encoder(visible_patches)
        
        # 3. 解码并重建
        full_tokens = self.generate_full_tokens(encoded, mask)
        reconstructed = self.decoder(full_tokens)
        
        # 4. 计算重建损失
        loss = self.reconstruction_loss(reconstructed, patches, mask)
        return loss