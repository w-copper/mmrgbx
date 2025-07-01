from mmengine.optim import OptimWrapper
from collections import deque
from mmrgbx.registry import OPTIM_WRAPPERS


@OPTIM_WRAPPERS.register_module()
class SkipWaveOptimWrapper(OptimWrapper):
    def __init__(self, acc_count=30, alpha=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acc_count = acc_count
        self.loss_queue = deque()
        self.alpha = alpha

    def update_params(self, loss, step_kwargs=None, zero_kwargs=None):
        if step_kwargs is None:
            step_kwargs = {}
        if zero_kwargs is None:
            zero_kwargs = {}
        loss = self.scale_loss(loss)
        if len(self.loss_queue) >= self.acc_count:
            mean_loss = sum(self.loss_queue) / self.acc_count
            if loss > mean_loss + self.alpha:
                loss_scale = 1.0 / loss.item() * 0.01 * mean_loss
                loss = loss * loss_scale
            else:
                self.loss_queue.popleft()
                self.loss_queue.append(loss.item())
        else:
            self.loss_queue.append(loss.item())

        self.backward(loss)
        # Update parameters only if `self._inner_count` is divisible by
        # `self._accumulative_counts` or `self._inner_count` equals to
        # `self._max_counts`
        if self.should_update():
            self.step(**step_kwargs)
            self.zero_grad(**zero_kwargs)
