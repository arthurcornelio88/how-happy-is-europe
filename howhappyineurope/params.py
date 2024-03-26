import json
import numpy as np
import pandas as pd

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

FEATURES_DICT = {
    "stfmjob":  "How satisfied are you in your main job",
    "trdawrk":  "Too tired after work to enjoy things like doing at home, how often",
    "jbprtfp":  "Job prevents you from giving time to partner/family, how often",
    "pfmfdjba": "Partner/family fed up with pressure of your job, how often",
    "dcsfwrka": "Current job: can decide time start/finish work",
    "wrkhome":  "Work from home or place of choice, how often",
    "wrklong":  "Employees expected to work overtime, how often",
    "wrkresp":  "Employees expected to be responsive outside working hours, how often",
    "health":   "Subjective general health",
    "stfeco":   "How satisfied with present state of economy in country",
    "hhmmb":    "Number of people living regularly as member of household",
    "hincfel":  "Feeling about household's income nowadays",
    "trstplc":  "Trust in the police",
    "sclmeet":  "How often socially meet with friends, relatives or colleagues",
    "hlthhmp":  "Hampered in daily activities by illness/disability/infirmity/mental problem",
    "sclact":   "Take part in social activities compared to others of same age",
    "iphlppl":  "Important to help people and care for others well-being",
    "ipsuces":  "Important to be successful and that people recognise achievements",
    "ipstrgv":  "Important that government is strong and ensures safety",
    "gndr"   :  "Gender",
    "cntry"  :  "Country",
    "happy":    "Happiness"
}

X_PRED = pd.DataFrame([np.array(["FR",1,1,1,1,1,6,1,1,1,1])], columns=['cntry', \
'gndr', 'sclmeet', 'inprdsc', 'sclact', 'health', 'rlgdgr','dscrgrp',     \
'ctzcntr', 'brncntr', 'happy'])

GAR_IMAGE="how-happy-in-europe"

with open("features_table.json", 'r') as file:
    FEATURE_TABLE = json.load(file)
