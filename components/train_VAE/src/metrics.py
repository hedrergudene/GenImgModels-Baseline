#
# Metrics
#


# Requirements
import torch
import torch.nn.functional as F
from typing import Callable, Sequence, Union

# SSIM (extracted from PyTorch Ignite: https://pytorch.org/ignite/generated/ignite.metrics.SSIM.html)
class SSIM(torch.nn.Module):
    """
    Computes Structual Similarity Index Measure

    Args:
        data_range: Range of the image. Typically, ``1.0`` or ``255``.
        kernel_size: Size of the kernel. Default: (11, 11)
        sigma: Standard deviation of the gaussian kernel.
            Argument is used if ``gaussian=True``. Default: (1.5, 1.5)
        k1: Parameter of SSIM. Default: 0.01
        k2: Parameter of SSIM. Default: 0.03
        gaussian: ``True`` to use gaussian kernel, ``False`` to use uniform kernel
        output_transform: A callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.
    """
    def __init__(
        self,
        data_range: Union[int, float],
        kernel_size: Union[int, Sequence[int]] = (11, 11),
        sigma: Union[float, Sequence[float]] = (1.5, 1.5),
        k1: float = 0.01,
        k2: float = 0.03,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cuda"),
        ):
        # Parameters
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]  # type: Sequence[int]
        elif isinstance(kernel_size, Sequence):
            self.kernel_size = kernel_size
        else:
            raise ValueError("Argument kernel_size should be either int or a sequence of int.")

        if isinstance(sigma, float):
            self.sigma = [sigma, sigma]  # type: Sequence[float]
        elif isinstance(sigma, Sequence):
            self.sigma = sigma
        else:
            raise ValueError("Argument sigma should be either float or a sequence of float.")

        if any(x % 2 == 0 or x <= 0 for x in self.kernel_size):
            raise ValueError(f"Expected kernel_size to have odd positive number. Got {kernel_size}.")

        if any(y <= 0 for y in self.sigma):
            raise ValueError(f"Expected sigma to have positive number. Got {sigma}.")

        super(SSIM, self).__init__()
        self.c1 = (k1 * data_range) ** 2
        self.c2 = (k2 * data_range) ** 2
        self.pad_h = (self.kernel_size[0] - 1) // 2
        self.pad_w = (self.kernel_size[1] - 1) // 2
        self._device = device
        self.__name__="SSIM"


    def forward(self,
                output:torch.Tensor,
                batch:torch.Tensor,
                )->float:
        y_pred, y = torch.sigmoid(output["recon_batch"].detach()), batch.detach()

        if y_pred.dtype != y.dtype:
            raise TypeError(
                f"Expected y_pred and y to have the same data type. Got y_pred: {y_pred.dtype} and y: {y.dtype}."
            )

        if y_pred.shape != y.shape:
            raise ValueError(
                f"Expected y_pred and y to have the same shape. Got y_pred: {y_pred.shape} and y: {y.shape}."
            )

        if len(y_pred.shape) != 4 or len(y.shape) != 4:
            raise ValueError(
                f"Expected y_pred and y to have BxCxHxW shape. Got y_pred: {y_pred.shape} and y: {y.shape}."
            )
        
        kernel = self._gaussian(kernel_size=self.kernel_size[0], sigma=self.sigma[0])
        channel = y_pred.size(1)
        if len(kernel.shape) < 4:
            kernel = kernel.expand(channel, 1, -1, -1).to(device=y_pred.device)

        y_pred = F.pad(y_pred, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")
        y = F.pad(y, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")

        input_list = torch.cat([y_pred, y, y_pred * y_pred, y * y, y_pred * y])
        outputs = F.conv2d(input_list, kernel, groups=channel)

        output_list = [outputs[x * y_pred.size(0) : (x + 1) * y_pred.size(0)] for x in range(len(outputs))]

        mu_pred_sq = output_list[0].pow(2)
        mu_target_sq = output_list[1].pow(2)
        mu_pred_target = output_list[0] * output_list[1]

        sigma_pred_sq = output_list[2] - mu_pred_sq
        sigma_target_sq = output_list[3] - mu_target_sq
        sigma_pred_target = output_list[4] - mu_pred_target

        a1 = 2 * mu_pred_target + self.c1
        a2 = 2 * sigma_pred_target + self.c2
        b1 = mu_pred_sq + mu_target_sq + self.c1
        b2 = sigma_pred_sq + sigma_target_sq + self.c2

        ssim_idx = (a1 * a2) / (b1 * b2)
        sum_of_batchwise_ssim = torch.mean(ssim_idx, (1, 2, 3), dtype=torch.float64).to(self._device)
        return torch.sum(sum_of_batchwise_ssim / y.shape[0])


    def _gaussian(self, kernel_size: int, sigma: float) -> torch.Tensor:
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, device=self._device)
        gauss = torch.exp(-0.5 * (kernel / sigma).pow(2))
        return (gauss / gauss.sum()).unsqueeze(dim=0)  # (1, kernel_size)
