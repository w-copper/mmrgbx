# Copyright (c) Open-CD. All rights reserved.
"""Open-CD provides 17 registry nodes to support using modules across projects.
Each node is a child of the root registry in MMEngine.
More details can be found at
https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
"""

from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmseg.registry import MODELS as MMSEG_MODELS, TRANSFORMS as MMSEG_TRANSFORMS
from mmengine.registry import Registry
from mmseg.registry import (
    DATA_SAMPLERS,
    EVALUATOR,
    HOOKS,
    INFERENCERS,
    LOG_PROCESSORS,
    LOOPS,
    METRICS,
    MODEL_WRAPPERS,
    OPTIM_WRAPPER_CONSTRUCTORS as MMSEG_OPTIM_WRAPPER_CONSTRUCTORS,
    OPTIM_WRAPPERS as MMSEG_OPTIM_WRAPPERS,
    OPTIMIZERS,
    PARAM_SCHEDULERS,
    RUNNER_CONSTRUCTORS,
    RUNNERS,
    TASK_UTILS,
    VISBACKENDS,
    VISUALIZERS,
    WEIGHT_INITIALIZERS,
)

# manage data-related modules
DATASETS = Registry("dataset", parent=MMENGINE_DATASETS, locations=["mmrgbx.datasets"])
TRANSFORMS = Registry(
    "transform", parent=MMSEG_TRANSFORMS, locations=["mmrgbx.datasets.transforms"]
)

MODELS = Registry("model", parent=MMSEG_MODELS, locations=["mmrgbx.models"])

OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    "optim_wrapper_constructor",
    parent=MMSEG_OPTIM_WRAPPER_CONSTRUCTORS,
    locations=["mmrgbx.optim"],
)

OPTIM_WRAPPERS = Registry(
    "optim_wrapper", parent=MMSEG_OPTIM_WRAPPERS, locations=["mmrgbx.optim"]
)
