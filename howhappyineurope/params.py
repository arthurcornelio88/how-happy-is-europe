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

X_PRED = pd.DataFrame([np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,"FR"])], \
    #X_PRED = pd.DataFrame([np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])], \
    columns=["stfmjob","trdawrk","jbprtfp", "pfmfdjba", "dcsfwrka", "wrkhome", \
        "wrklong", "wrkresp", "health","stfeco","hhmmb","hincfel", "trstplc", \
        "sclmeet", "hlthhmp", "sclact","iphlppl", "ipsuces", "ipstrgv", "gndr", "cntry"])\

    #    "cntry_GB", "cntry_CZ", "cntry_SI", "cntry_FI", "cntry_BG", "cntry_PT", \
    #        "cntry_NL", "cntry_LT", "cntry_IE", "cntry_HU", "cntry_BE", "cntry_ME", \
    #            "cntry_NO", "cntry_CH", "cntry_GR", "cntry_SK", "cntry_HR", \
    #                "cntry_IS", "cntry_MK", "cntry_EE", "cntry_IT"])

GAR_IMAGE="how-happy-in-europe"

with open("features_table.json", 'r') as file:
    FEATURE_TABLE = json.load(file)
