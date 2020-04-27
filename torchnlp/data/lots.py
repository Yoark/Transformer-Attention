from torchtext.data.iterator import Iterator, BucketIterator
from torchtext.data.field import RawField
import torch
# class AttnIterator(BucketIterator):
#     def __init__(self, dataset, batch_size, sort_key=None, device=None,
#                  batch_size_fn=None, train=True,
#                  repeat=False, shuffle=None, sort=None,
#                  sort_within_batch=None, adv=False):
#         super().__init__(self, dataset, batch_size, sort_key=None, device=None,
#                  batch_size_fn=None, train=True,
#                  repeat=False, shuffle=None, sort=None,
#                  sort_within_batch=None)
        
#         if adv:
            
class AttnField(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, is_target=False):
        super().__init__(preprocessing=preprocessing, postprocessing=postprocessing, is_target=is_target)

    def process(self, batch, *args, **kwargs):
        new_batch = []
        for item in batch:
            new_batch.append(torch.FloatTensor(item))
        return new_batch

class PredField(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, is_target=False):
        super().__init__(preprocessing=preprocessing, postprocessing=postprocessing, is_target=is_target)

    def process(self, batch, *args, **kwargs):
        new_batch = []
        for item in batch:
            new_batch.append(torch.FloatTensor(item))
        return new_batch