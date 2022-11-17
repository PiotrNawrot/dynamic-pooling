import torch
from utils.exp_utils import init_seed


def autoregressive_test(model, device):
    """
        Assumptions:
            Vocab size >= 10
    """
    target_test_len = 32
    input_data = torch.randint(low=0, high=10, size=(target_test_len + 1, 1)).to(device)

    # target = input_data[1:]
    input_data = input_data[:-1]

    boundaries = torch.zeros_like(input_data, dtype=torch.bool)
    boundaries[:, ::2] = 1

    model.eval()

    with torch.no_grad():
        init_seed(0)
        full_logits = model(input_data, None, boundaries)

        for i in range(target_test_len):
            init_seed(0)
            last_logit = model(input_data[:i + 1], None, boundaries[:i + 1])[-1]
            assert torch.allclose(last_logit, full_logits[i], atol=1e-6)

    print('The model passed the autoregresivity test')

    model.train()
