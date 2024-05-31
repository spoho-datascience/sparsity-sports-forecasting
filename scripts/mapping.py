import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

from scripts.helper import get_team_map

### Mapping footballdata leagues and seasons to training data format ###
path_fbd = "C:\\Users\\ke6564\\Desktop\\Studium\\Promotion\\PythonProjects\\2023-soccer-prediction\\input\\training_data\\all_leagues.csv"
football_data = pd.read_csv(path_fbd, dtype={"Season": str})


football_data["Date"] = pd.to_datetime(football_data["Date"], dayfirst=True)
fbd_filt = football_data.loc[football_data["Date"] >= "2000-01-01"]
fbd_filt.loc[pd.isna(fbd_filt["comp_div"]) & (fbd_filt["Div"] == "Super League"), "comp_div"] = "CHN1"
fbd = fbd_filt.loc[(pd.isna(fbd_filt["comp_div"]) == False) & (pd.isna(fbd_filt["HomeTeam"]) == False)]

season_mapping = {
    '0001': "00-01",
    '0102': "01-02",
    '0203': "02-03",
    '0304': "03-04",
    '0405': "04-05",
    '0506': "05-06",
    '0607': "06-07",
    '0708': "07-08",
    '0809': "08-09",
    '0910': "09-10",
    '1011': "10-11",
    '1112': "11-12",
    '1213': "12-13",
    '1314': "13-14",
    '1415': "14-15",
    '1516': "15-16",
    '1617': "16-17",
    '1718': "17-18",
    '1819': "18-19",
    '1920': "19-20",
    '2021': "20-21",
    '2122': "21-22",
    '2223': "22-23",
    '2012/2013': "12-13",
    '2013/2014': "13-14",
    '2014/2015': "14-15",
    '2015/2016': "15-16",
    '2016/2017': "16-17",
    '2017/2018': "17-18",
    '2018/2019': "18-19",
    '2019/2020': "19-20",
    '2020/2021': "20-21",
    '2021/2022': "21-22",
    '2022/2023': "22-23",
    '2012': "12-13",
    '2013': "13-14",
    '2014': "14-15",
    '2015': "15-16",
    '2016': "16-17",
    '2017': "17-18",
    '2018': "18-19",
    '2019': "19-20",
    '2020': "20-21",
    '2022': "22-23",
}

fucked_leagues = ["ARG1", "BRA1", "CHN1", "FIN1", "JPN1", "NOR1", "SWE1", "USA1"]

fbd["standard_season"] = fbd["Season"]
fbd.loc[fbd["comp_div"].isin(fucked_leagues) & (fbd["standard_season"] == "2021"), "standard_season"] = "2021/2022"

fbd["Sea"] = fbd["standard_season"].replace(season_mapping)

fbd.to_csv("fbd_mapped.csv", index=False)
#########################################

fbd = pd.read_csv("fbd_mapped.csv")
training_data = pd.read_csv("input/training_data/training_df.csv")

groups = list(training_data.groupby(["Sea", "Lge"]).groups.keys())


sea, lge = ("14-15", "ARG1")


super_map = []
for sea, lge in groups:
    season_fbd = fbd.loc[(fbd["Sea"] == sea) & (fbd["comp_div"] == lge)]
    season_train = training_data.loc[(training_data["Sea"] == sea) & (training_data["Lge"] == lge)]

    team_map, converged = get_team_map(season_train, season_fbd)

    if converged in ["Partially", "Fully"]:
        super_map += [{"sea": sea, "lge": lge, "team_train": k, "team_fbd": v, "converged": converged} for k, v in team_map.items()]
    else:
        super_map += [{"sea": sea, "lge": lge, "converged": converged}]


gladbach = [x for x in super_map if x["team_train"] == "Monchengladbach"]
chinese = [x for x in super_map if x["lge"] == "CHN1"]
not_converged = [x for x in super_map if x["converged"] == "No matches"]

super_map_df = pd.DataFrame(super_map)

super_map_df.to_json("supermap.jsonl", orient="records", lines=True)

sco1 = fbd.loc[fbd[("comp_div"] == "SCO3") & ]

no_matches = [x for x in super_map if x["converged"] == "No matches"]

empty = [x for x in super_map if x["converged"] == "Empty"]
partially = [x for x in super_map if x["converged"] == "Partially"]



ger_map =