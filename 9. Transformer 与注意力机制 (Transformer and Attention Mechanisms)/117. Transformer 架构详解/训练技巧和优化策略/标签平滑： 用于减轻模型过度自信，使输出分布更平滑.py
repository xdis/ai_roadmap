def label_smoothed_nll_loss(lprobs, target, epsilon=0.1):
    # 实现标签平滑的交叉熵损失
    target = target.unsqueeze(-1)  # 添加维度以匹配lprobs
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss.sum()