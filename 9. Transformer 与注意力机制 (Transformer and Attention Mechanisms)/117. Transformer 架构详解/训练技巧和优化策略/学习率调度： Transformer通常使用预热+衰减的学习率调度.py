def get_lr(step, d_model, warmup_steps=4000):
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))