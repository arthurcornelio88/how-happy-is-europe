import os
import numpy as np
import pandas as pd

##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")

STATE_OF_HAPPINESS = [
    "Extremely unhappy",
    "Extremely unhappy",
    "Extremely unhappy",
    "Extremely unhappy",
    "Neutral",
    "Neutral",
    "Neutral",
    "Neutral",
    "Extremely happy",
    "Extremely happy",
    "Extremely happy"
]

X_PRED = pd.DataFrame([np.array(["FR",1,1,1,1,1,6,1,1,1,1])], columns=['cntry', \
'gndr', 'sclmeet', 'inprdsc', 'sclact', 'health', 'rlgdgr','dscrgrp',     \
'ctzcntr', 'brncntr', 'happy'])
