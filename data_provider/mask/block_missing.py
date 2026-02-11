import numpy as np
import torch


def generate_block_missing_mask(
    data: torch.Tensor, miss_rate: float, block_len=10, block_width=1
) -> torch.Tensor:
    n_steps, n_features = data.shape

    num_blocks = int(
        miss_rate * n_steps * n_features // (block_len * block_width)
    )
    mask = torch.ones_like(data)
    for _ in range(num_blocks):
        while True:
            start_step = np.random.randint(0, n_steps - block_len + 1)
            start_feature = np.random.randint(0, n_features - block_width + 1)
            if torch.all(
                mask[
                    start_step : start_step + block_len,
                    start_feature : start_feature + block_width,
                ]
                == 1
            ):
                mask[
                    start_step : start_step + block_len,
                    start_feature : start_feature + block_width,
                ] = 0
                break

    return mask


def generate_block_missing_mask_at_random(
    data: torch.Tensor, miss_rate: float, max_block_len=10, max_block_width=1
) -> torch.Tensor:
    n_steps, n_features = data.shape
    total_elements = n_steps * n_features
    num_missing_elements = int(miss_rate * total_elements)
    mask = torch.ones_like(data)

    missing_elements = 0

    while missing_elements < num_missing_elements:
        block_len = np.random.randint(1, max_block_len + 1)
        block_width = np.random.randint(1, max_block_width + 1)
        start_step = np.random.randint(0, n_steps - block_len + 1)
        start_feature = np.random.randint(0, n_features - block_width + 1)

        current_block_size = block_len * block_width

        if torch.all(
            mask[
                start_step : start_step + block_len,
                start_feature : start_feature + block_width,
            ]
            == 1
        ):
            mask[
                start_step : start_step + block_len,
                start_feature : start_feature + block_width,
            ] = 0
            missing_elements += current_block_size

    return mask


if __name__ == "__main__":
    data = torch.randn(10000, 320)
    miss_rate = 0.3
    mask = generate_block_missing_mask_at_random(data, miss_rate, 20, 5)
    print(mask.sum() / mask.numel())
