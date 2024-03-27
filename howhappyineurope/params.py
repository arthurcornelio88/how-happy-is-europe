import json
import numpy as np
import pandas as pd

GAR_IMAGE="how-happy-in-europe"
ROOT_DIR = "howhappyineurope"

#STATE_OF_HAPPINESS[0] : unhappy
#STATE_OF_HAPPINESS[1] : neutral
#STATE_OF_HAPPINESS[2] : happy
STATE_OF_HAPPINESS = [
    "unhappy...",
    "neutral.",
    "happy!",
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


CONT_COLS = ["stfmjob","trdawrk","jbprtfp", "pfmfdjba", "dcsfwrka", "wrkhome", \
        "wrklong", "wrkresp", "health","stfeco","hhmmb","hincfel", "trstplc", \
        "sclmeet", "hlthhmp", "sclact","iphlppl", "ipsuces", "ipstrgv", "gndr"]
CATEG_COLS = ["cntry"]

X_PRED = pd.DataFrame(
    data=[
        np.array(
            [1,1,1,1,1,
            1,1,1,1,1,
            1,1,1,1,1,
            1,1,1,1,1,
            "BE"]
        )
    ],
    columns=CONT_COLS + CATEG_COLS)

for column in X_PRED.columns:
    if column != 'cntry':
        X_PRED[column] = X_PRED[column].astype(int)

with open("data/features_table.json", 'r') as file:
    FEATURES_TABLE = json.load(file)
