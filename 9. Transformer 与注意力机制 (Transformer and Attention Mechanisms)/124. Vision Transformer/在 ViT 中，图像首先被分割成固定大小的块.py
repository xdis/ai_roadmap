def patch_images(images, patch_size):
    """将批量图像分割成块
    
    Args:
        images: 形状为 [B, C, H, W] 的张量
        patch_size: 块的大小（例如 16 表示 16x16 像素）
        
    Returns:
        形状为 [B, N, C*P*P] 的张量，其中 N = HW/P²，P 是块大小
    """
    B, C, H, W = images.shape
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(B, -1, C * patch_size * patch_size)
    return patches