import torch

from llmcompressor.observers.base import Observer
from llmcompressor.observers.helpers import flatten_for_calibration

__all__ = ["HistogramObserver"]


@Observer.register("histogram")
class HistogramObserver(Observer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        observer_kwargs = self.args.observer_kwargs
        self.bins = observer_kwargs.get("bins", 2048)
        self.histogram = None
        self.xmax = -torch.inf

    @torch.no_grad
    def forward(self, observed: torch.Tensor) -> ScaleZpTuple:
        """
        Calculate updated scales and zero points from observed value
        (weight, activation, or attention state).

        :param observed: value being observed
        :return: calibrated scale and zero point
        """
        self._add_batch(observed)
        return (None, None)

    def _add_batch(self, observed: torch.Tensor):
        g_idx = self._get_module_param("g_idx")
        observed = flatten_for_calibration(observed, self.base_name, self.args, g_idx)
        xmin, xmax = torch.aminmax(observed, dim=[0, 1])  #
        xmax = xmax * (xmax > -xmin) - xmin(xmax <= -xmin)

        new_max = torch.max(self.xmax, xmax)

        if self.histogram is None:
            return
            # init histogram
        # hist = torch.histdd(observed,

        # need the shape to have groups as final dimension for histogramdd

        # probably just need to store the max of all the bins

        # should probably just inherit from minmax observer and steal
        # that functionality and tack this functionality on at the end
