# Copyright (c) Open-CD. All rights reserved.
import mmcv
import mmengine
from mmengine.utils import digit_version

import mmseg
from .version import __version__, version_info

mmcv_minimum_version = "2.0.0rc4"
mmcv_maximum_version = "2.3.0"
mmcv_version = digit_version(mmcv.__version__)

mmengine_minimum_version = "0.6.0"
mmengine_maximum_version = "1.0.0"
mmengine_version = digit_version(mmengine.__version__)

mmseg_minimum_version = "1.0.0rc6"
mmseg_maximum_version = "1.3.0"
mmseg_version = digit_version(mmseg.__version__)


assert mmcv_version >= digit_version(
    mmcv_minimum_version
) and mmcv_version < digit_version(mmcv_maximum_version), (
    f"MMCV=={mmcv.__version__} is used but incompatible. "
    f"Please install mmcv>={mmcv_minimum_version}, <{mmcv_maximum_version}."
)

assert mmengine_version >= digit_version(
    mmengine_minimum_version
) and mmengine_version < digit_version(mmengine_maximum_version), (
    f"MMEngine=={mmengine.__version__} is used but incompatible. "
    f"Please install mmengine>={mmengine_minimum_version}, "
    f"<{mmengine_maximum_version}."
)

assert mmseg_version >= digit_version(
    mmseg_minimum_version
) and mmseg_version < digit_version(mmseg_maximum_version), (
    f"MMSegmentation=={mmseg.__version__} is used but incompatible. "
    f"Please install mmseg>={mmseg_minimum_version}, "
    f"<{mmseg_maximum_version}."
)


__all__ = ["__version__", "version_info", "digit_version"]
