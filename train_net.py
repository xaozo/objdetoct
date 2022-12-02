#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader, DatasetMapper
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.engine.hooks import EvalHook
from detectron2.evaluation import (
    COCOEvaluator,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

from project_package.config import add_config
from LossEvalHook import LossEvalHook
#from early_stop import early_stopping


class EvalHook(EvalHook):
    """
    Override EvalHook to implement early stopping
    1. Override _do_eval and after_step methods.
     - The hooks are executed after_step. The EvalHook will return a signal. If the signal is True, we break the loop.
    
    def __init__(self,eval_period, eval_function):
        super(EvalHook,self).__init__(eval_period, eval_function)
        
    def _do_eval(self):
        result = self._func()
        return result
        
    def after_step(self):
        signal = None
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            # do the last eval in after_train
            if next_iter != self.trainer.max_iter:
                signal = self._do_eval()
        return signal
    """

class Trainer(DefaultTrainer):
    """
    Override DefaultTrainer functions to implement:
    1. Early stopping
    2. Validation loss tracking
    
    ########## EARLY STOPPING ##########
    def __init__(self,cfg):
        super(DefaultTrainer,self).__init__()
        self.patience_iter = 0
        
    def train(self, start_iter: int, max_iter: int):

        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    signal = self.after_step()
                    if signal:
                        break
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()
                
    def after_step(self):
        signal=None
        for h in self._hooks:
            signal = h.after_step()
        return signal
    ####################################
    """
        
    ########## VALIDATION LOSS ##########
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks
    #####################################
    

    
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    add_config(cfg) # ADD CONFIG
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
