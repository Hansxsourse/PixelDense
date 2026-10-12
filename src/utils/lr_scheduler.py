from transformers import get_constant_schedule_with_warmup

class ConstantWithWarmup:
    def __init__(self, num_warmup_steps: int):
        self.num_warmup_steps = num_warmup_steps

    def __call__(self, optimizer):
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps
        )