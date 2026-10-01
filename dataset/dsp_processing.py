"""
Deprecated duplicate (v1 kept a second copy of the DSP code here).

The single implementation lives in preprocessing/dsp_processing.py;
this module only re-exports it so old imports keep working.
"""

from preprocessing.dsp_processing import *  # noqa: F401,F403
