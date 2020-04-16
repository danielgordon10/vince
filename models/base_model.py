import os

from dg_util.python_utils import pytorch_util as pt_util

import constants


class BaseModel(pt_util.BaseModel):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args

    def restore(self, skip_filter=None) -> int:
        iteration = 0
        if self.args.restore:
            iteration = super(BaseModel, self).restore(
                self.args.checkpoint_dir, self.args.saved_variable_prefix, self.args.new_variable_prefix, skip_filter
            )
        return iteration

    def save(self, iteration, num_to_keep=1):
        if self.args.save:
            pt_util.save(self, os.path.join(self.args.checkpoint_dir, constants.TIME_STR), num_to_keep, iteration)
            if self.saves > 0 and self.saves % self.args.long_save_frequency == 0:
                pt_util.save(self, self.args.long_save_checkpoint_dir, -1, iteration)
            self.saves += 1
