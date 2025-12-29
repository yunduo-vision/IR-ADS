from torch import nn
from torch.optim import AdamW, SGD


def get_optimizer(model: nn.Module, optimizer: str, lr: float, train_type: str, weight_decay: float = 0.01):
    wd_params, nwd_params = [], []
    if 'Adapter' in train_type:
        print("Only training adapter parameters. They are: ")

        params = [
            {"params": [p for n, p in model.named_parameters() if
                        ("Adapter" in n and p.requires_grad) or ("extra_patch_embed" in n and p.requires_grad) or (
                                    "head" in n and p.requires_grad) or (
                                    "MPG" in n and p.requires_grad)]}
        ]

        for n, p in model.named_parameters():

            if "Adap" not in n and "extra_patch_embed" not in n and "head" not in n and "MPG" not in n:  #
                p.requires_grad = False
            else:
                print(n)

        total_num = sum(p.numel() for n, p in model.named_parameters())
        trainable_num = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
        print('=Total: ', total_num, 'Trainable: ', trainable_num)

        for n, p in model.named_parameters():
            if p.requires_grad:
                print(n, p.numel())


    else:
        for p in model.parameters():
            if p.requires_grad:
                if p.dim() == 1:
                    nwd_params.append(p)
                else:
                    wd_params.append(p)

        params = [
            {"params": wd_params},
            {"params": nwd_params, "weight_decay": 0}
        ]

    if optimizer == 'adamw':
        return AdamW(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    else:
        return SGD(params, lr, momentum=0.9, weight_decay=weight_decay)