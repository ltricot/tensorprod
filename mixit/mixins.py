from . import interfaces

from ..utils import totest


@totest
class Snapshot(
    interfaces.Trainable,
    interfaces.Saveable,
):  # this might prevent MRO construction (if Trainable implements training)
    """Implements a regularly saving train loop."""
    def training(self, limit=None, step=-1, every=100):
        for step in iter(lambda: step + 1, None):
            if (step + 1) % every == 0:
                self.save()
            yield self.run_train()
