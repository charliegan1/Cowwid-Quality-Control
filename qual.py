# %% Libraries and Directories 

import openpyxl
import pandas as pd
import os
from pathlib import Path
from glob import iglob
from pathlib import Path
from datetime import date, datetime, timedelta
import numpy as np
from bokeh.models import ColumnDataSource, Whisker, BasicTicker, ColorBar, LinearColorMapper, PrintfTickFormatter, Span
from bokeh.plotting import figure, show, save
from bokeh.transform import factor_cmap
from bokeh.models import Range1d, Div, SingleIntervalTicker
from bokeh.io import output_file, show
from bokeh.layouts import gridplot, row, layout
from calendar import MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SUNDAY
from scipy.stats import linregress

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# %% Define path of backup folder, Q drive, encryption key, and N2 data
path_of_the_directory_charlie = ('/Users/charlesgan/Library/Mobile Documents/'
                                 'com~apple~CloudDocs/Eawag Covid Work/ExcelResults_backup')
                                 
path_of_the_directory = ('/Volumes/PCR_Cowwid/01_dPCR_data/01_Stilla/NCX_ExpandedSampling/ExcelResults')

path_N2 = ('/Volumes/PCR_Cowwid/01_dPCR_data/01_Stilla/NCX_GrippeAssay/RealWorldData/ExcelResults/LatestFluData.csv')

# %%Toggle which code to be run
status = 'real' # 'testing' or 'real'
status_data = 'recent' # 'recent' or 'all data'

# %% Run Covid Data Sender to gather recent conc. data
# Covid Data Sender
# Toggle for full data set (1 is full, 0 is only past week)
full = 0

# Display settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Define which treatment plants and their corresponding links to take from
flow_dict = {"05_1": "https://sensors-eawag.ch/sars/__data__/processed_normed_data_lugano_v1.csv",
            "05_2": "https://sensors-eawag.ch/sars/__data__/processed_normed_data_lugano_v2.csv",
            "10_1": "https://sensors-eawag.ch/sars/__data__/processed_normed_data_zurich_v1.csv",
            "10_2": "https://sensors-eawag.ch/sars/__data__/processed_normed_data_zurich_v2.csv",
            "12_1": "https://sensors-eawag.ch/sars/__data__/processed_normed_data_lausanne.csv",
            "16_1": "https://sensors-eawag.ch/sars/__data__/processed_normed_data_geneve_v1.csv",
            "16_2": "https://sensors-eawag.ch/sars/__data__/processed_normed_data_geneve_v2.csv",
            "17_1": "https://sensors-eawag.ch/sars/__data__/processed_normed_data_chur_v1.csv",
            "17_2": "https://sensors-eawag.ch/sars/__data__/processed_normed_data_chur_v2.csv",
            "19_1": "https://sensors-eawag.ch/sars/__data__/processed_normed_data_altenrhein_v1.csv",
            "19_2": "https://sensors-eawag.ch/sars/__data__/processed_normed_data_altenrhein_v2.csv",
            "25_1": "https://sensors-eawag.ch/sars/__data__/processed_normed_data_laupen_v1.csv",
            "25_2": "https://sensors-eawag.ch/sars/__data__/processed_normed_data_laupen_v2.csv"}

# Create empty data DataFrame
df = pd.DataFrame()

# Loop through each treatment plant and add to empty DataFrame
for key in flow_dict:
    df0 = pd.read_csv(flow_dict[key],sep = ";")
    df0["WWTP"] = key
    df0.rename(columns={ 'Unnamed: 0': 'sample_date' }, inplace = True)
    df0["sample_date"] = pd.to_datetime(df0["sample_date"])
    df0["WWTP_ID"] = "-"

    if "05" in df0["WWTP"][1]:
        df0["ARA_name"] = "CDA_Lugano"
        df0["ARA_email"] = "lorenzo.balmelli@cdaled.ch"
        df0["WWTP_ID"] = "05"
        df0["ARA_ID"] = 515100
        df0["pop"] = 124000          # goog_sheet = 124990 (christian value used instead)
    elif "10" in df0["WWTP"][1]:
        df0["ARA_name"] = "ARA_Werdhölzli"
        df0["ARA_email"] = "rey.eyer@zuerich.ch"
        df0["WWTP_ID"] = "10"
        df0["ARA_ID"] = 26101
        df0["pop"] = 471000          # goog_sheet = 429371 (christian value used instead)
    elif "12" in df0["WWTP"][1]:
        df0["ARA_name"] = "STEP_Vidy"
        df0["ARA_email"] = "eau@lausanne.ch"
        df0["WWTP_ID"] = "12"
        df0["ARA_ID"] = 558600
        df0["pop"] = 240000          # goog_sheet = 235359 (christian value used instead)
    elif "16" in df0["WWTP"][1]:
        df0["ARA_name"] = "STEP_Aire"
        df0["ARA_email"] = "Axel.Wahl@sig-ge.ch"
        df0["WWTP_ID"] = "16"
        df0["ARA_ID"] = 664301
        df0["pop"] = 451771          # goog_sheet = 440750 (christian value used instead)
    elif "17" in df0["WWTP"][1]:
        df0["ARA_name"] = "ARA_Chur"
        df0["ARA_email"] = "curdin.hedinger@chur.ch"
        df0["WWTP_ID"] = "17"
        df0["ARA_ID"] = 390101
        df0["pop"] = 55000           # goog_sheet = 54610 (christian value used instead)
    elif "19" in df0["WWTP"][1]:
        df0["ARA_name"] = "ARA_Altenrhein"
        df0["ARA_email"] = "hansruedi.graf@ava-altenrhein.ch"
        df0["WWTP_ID"] = "19"
        df0["ARA_ID"] = 323700
        df0["pop"] = 64000           # goog_sheet = 55968 (christian value used instead)
    elif "25" in df0["WWTP"][1]:
        df0["ARA_name"] = "ARA_Laupen"
        df0["ARA_email"] = "geschaeftsleitung.ara@sensetal.ch"
        df0["WWTP_ID"] = "25"
        df0["ARA_ID"] = 66700
        df0["pop"] =  62000          # goog_sheet = 57779 (christian value used instead)
    else:
        print("ERROR - WWTP not found")

    df = pd.concat([df,df0], axis = 0)

# Segment a Dataframe with info on missing case and flow data
df_case_flow = df

# Selecting only the columns we need
df = df[["sample_date","WWTP_ID","ARA_ID","sars_cov2_rna [gc/(d*100000 capita)]","flow [m^3/d]",
         "quantification_flag [{Q: >LOQ,D: >LOD,N: <LOD}]","pop","ARA_name","ARA_email"]]

# Dropping all the rows with no SARS gc/L data and resetting the index
df = df.dropna()
df = df.reset_index()
df["flow [m^3/d]"] = df["flow [m^3/d]"].astype(float)

# Creating the sample name used internally and the sample date according to BAG requirements
df["sample_name"] = df["WWTP_ID"] + "_" + df["sample_date"].dt.strftime("%Y_%m_%d")
df["sample_date_format"] = df["sample_date"].dt.strftime("%Y-%m-%d")

# Conversion of [gc/d*100000 capita] to [gc/L]
df["sars"] = df["sars_cov2_rna [gc/(d*100000 capita)]"]/100000 # [gc/d   *population]
df["sars"] = df["sars"]/df["flow [m^3/d]"]                     # [gc/m^3 *population]
df["sars"] = df["sars"] / 1000                                 # [gc/L   *population]
df["sars"] = df["sars"]*df["pop"]                              # [gc/L]

# Segment a Dataframe with info on missing case and flow data
df_case_flow = df_case_flow.groupby(by = ['sample_date', 'ARA_name','WWTP_ID']).mean()
df_case_flow = df_case_flow.reset_index()
df_case_flow["flow [m^3/d]"] = df_case_flow["flow [m^3/d]"].astype(float)
df_case_flow["sample_name"] = df_case_flow["WWTP_ID"] + "_" + df_case_flow["sample_date"].dt.strftime("%Y_%m_%d")
df_case_flow["sample_date_format"] = df_case_flow["sample_date"].dt.strftime("%Y-%m-%d")
recent_day = max(df["sample_date"]) # can change here
delta_week = timedelta(days=7) # can change here
start_date = recent_day - delta_week
mask = (df_case_flow['sample_date'] >= start_date) & (df_case_flow['sample_date'] <= recent_day)
df_case_flow = df_case_flow[mask]

# Verify with google sheet (sensetal 29.03.2022)
if full == 1:
    df["sars_check_goog"] = df["sars"] / 1000          # [gc/mL WW]
    df["sars_check_goog"] = df["sars_check_goog"] * 40 # [gc/WW sample]
    df["sars_check_goog"] = df["sars_check_goog"] / 80 # [gc/uL sample]
    df["sars_check_goog"] = df["sars_check_goog"] * 5  # [gc/ul reaction]

# Define LOQ and Find LOQ LOD
df_loq_lod = df[['sample_date','sample_name', 'quantification_flag [{Q: >LOQ,D: >LOD,N: <LOD}]']]
mask = (df_loq_lod['sample_date'] >= start_date) & (df_loq_lod['sample_date'] <= recent_day)
df_loq_lod = df_loq_lod[mask]

df["LOQ"] = 27.4 # [gc/reaction]

# Index last weeks data for continual weekly updates
recent_day = max(df["sample_date"]) # can change here
delta_week = timedelta(days=300) # can change here
start_date = recent_day - delta_week
mask = (df['sample_date'] >= start_date) & (df['sample_date'] <= recent_day)

# Export to existing excel sheet requested by BAG

###### Sheet 1  [ARA_ID;ARA_name;ARA_email;ARA_popsize]
df.rename(columns = {'pop':'ARA_popsize'}, inplace = True)
df_meta_1 = df.drop_duplicates(subset = ["ARA_ID"])
df_meta_1 = df_meta_1[["ARA_ID","ARA_name","ARA_email","ARA_popsize"]]

###### Sheet 2  [Lab_name;Lab_email;Lab_phone]
df_meta_1["Lab_name"] = "EAWAG"
df_meta_1["Lab_email"] = "abwasser.covid@eawag.ch"
df_meta_1["Lab_phone"] = "+41587655632"
df_meta_2 = df_meta_1.drop_duplicates(subset = ["Lab_name"])
df_meta_2 = df_meta_2[["Lab_name","Lab_email","Lab_phone"]]

###### Sheet 3  [input_vol_proc;conc_method;extraction_kit;combi_kit;rna_elution_vol;pcr_rna_vol;pcr_kit;replicate_n]
df_meta_2["input_vol_proc"] = 40
df_meta_2["conc_method"] = "Wizard_Enviro_Total_Nucleic_Acid_Kit"
df_meta_2["extraction_kit"] = "Wizard_Enviro_Total_Nucleic_Acid_Kit"
df_meta_2["combi_kit"] = "Wizard_Enviro_Total_Nucleic_Acid_Kit"
df_meta_2["rna_elution_vol"] = 80
df_meta_2["pcr_rna_vol"] = 5
df_meta_2["pcr_kit"] = "qScript_XLT_OneStep_RTqPCR_ToughMix"
df_meta_2["replicate_n"] = 2

# Select out the necessary elements
df_meta_3 = df_meta_2[["input_vol_proc","conc_method","extraction_kit","combi_kit","rna_elution_vol","pcr_rna_vol","pcr_kit","replicate_n"]]
df_meta_2 = df_meta_2[["Lab_name","Lab_email","Lab_phone"]]
df_meta_1 = df_meta_1[["ARA_ID","ARA_name","ARA_email","ARA_popsize"]]

###### Sheet 4  [ARA_ID;sample_ID ;date_collection;Flow;conc_cov_target_N1_N2_E_M_S;LOQ]
df.rename(columns = {'sars':'conc_cov_target_N1_N2_E_M_S', 'flow [m^3/d]':'Flow','sample_name':'sample_ID',
                    'sample_date_format':'date_collection'}, inplace = True)
if full == 1:
    df_meta_4 = df[["ARA_ID","sample_ID","date_collection","Flow","conc_cov_target_N1_N2_E_M_S","LOQ"]]

else:
    recent_df = df[mask]
    df_meta_4 = recent_df[["ARA_ID","sample_ID","date_collection","Flow","conc_cov_target_N1_N2_E_M_S","LOQ"]]

# Export 4 files as csv
if full == 1:
    now = datetime.now()
    now = now.strftime("%Y%m%d")
    meta = 'ARA'
    dir = '/Volumes/EA-Daten/Messdaten/PCR_Cowwid/01_dPCR_data/01_Stilla/NCX_ExpandedSampling/Exporter/'

    pathname1 = f'{dir}{now}_EAWAG_{meta}.csv'
    df_meta_1.to_csv(path_or_buf = pathname1, sep = ';', header = True, index = False)

    meta = 'Lab'
    pathname2 = f'{dir}{now}_EAWAG_{meta}.csv'
    df_meta_2.to_csv(path_or_buf = pathname2, sep = ';', header = True, index = False)

    meta = 'Method'
    pathname3 = f'{dir}{now}_EAWAG_{meta}.csv'
    df_meta_3.to_csv(path_or_buf = pathname3, sep = ';', header = True, index = False)

    meta = 'Results'
    pathname4 = f'{dir}{now}_EAWAG_{meta}.csv'
    df_meta_4.to_csv(path_or_buf = pathname4, sep = ';', header = True, index = False)
else:
    now = datetime.now()
    now = now.strftime("%Y%m%d")
    meta = 'Results'
    dir = '/Volumes/BAG_Data/' # maybe change
    pathname4 = f'{dir}{now}_EAWAG_{meta}.csv'
    df_meta_4.to_csv(path_or_buf = pathname4, sep = ';', header = True, index = False)

#%% Merge N2 data to N1 dataframe

N2_conc = pd.DataFrame()
N2_conc = pd.read_csv(path_N2)
N2_conc['sample_date'] = pd.to_datetime(N2_conc['sample_date'])
N2_conc.sort_values(by=['sample_date'], inplace=True)

#%% Toggles for which directory to take from, locally (testing) or network (real)
if status == 'testing' :
    # Save all .xlsx files paths and modification time into paths
    paths = [(p.stat().st_mtime, p) for p in Path(path_of_the_directory_charlie).iterdir() if p.suffix == ".xlsx"]

    # Sort them by the modification time
    paths = sorted(paths, key=lambda x: x[0], reverse=True)

    # Get the last modified file and unlock
    last = str(paths[0][1])
    unlock(last)
    print(last)

if status == 'real' :
    # Save all .xlsx files paths and modification time into paths
    paths = [(p.stat().st_mtime, p) for p in Path(path_of_the_directory).iterdir() if p.suffix == ".xlsx"]

    # Sort them by the modification time
    paths = sorted(paths, key=lambda x: x[0], reverse=True)

    # Get the last modified file and unlock
    last = str(paths[0][1])
# %% Compile most recent compiled results file available
last_name = last[-15:-5:1]
print(last_name)

if status_data == 'all data' : 
# Create DFs using master sheet with all data
    df6 = pd.DataFrame
    df5 = pd.DataFrame
    df4 = pd.DataFrame
    df3 = pd.DataFrame
    df2 = pd.DataFrame
    df6 = pd.read_excel(last, sheet_name='SixthWave')
    df5 = pd.read_excel(last, sheet_name='FifthWave')
    df4 = pd.read_excel(last, sheet_name='FourthWave')
    df3 = pd.read_excel(last, sheet_name='ThirdWave')
    df2 = pd.read_excel(last, sheet_name='SecondWave')
    # Combine all data into one df and print to csv
    frames = [df6, df5, df4, df3, df2]
    result = pd.concat(frames)
    result.to_csv('data_all.csv')

if status_data == 'recent' : 
    df6 = pd.DataFrame 
    df6 = pd.read_excel(last, sheet_name='SixthWave')
    # Print to csv
    df6.to_csv('recent.csv')
    
# Load all data to dataframe to create source for dashboard
if status_data == 'all data' : 
    df = pd.read_csv('data_all.csv')
if status_data == 'recent' : 
    df = df6
#%% Calculate the failure rate and mask all the time frames of interest
df["sample_date"] = pd.to_datetime(df["sample_date"])
df["run_date"] = pd.to_datetime(df["run_date"])
df['failure'] = np.where(df['TotalDroplets']<=15000, 1, 0)
df['chip_counts'] = df.groupby(['PCR_ID', 'run_date'])['failure'].transform('count')
df['fail_counts'] = df.groupby(['PCR_ID', 'run_date'])['failure'].transform('sum')
df['failure_rate'] = df['fail_counts']/df['chip_counts']
df_all = df

# Mask week
recent_day = max(df["run_date"]) 
delta_week = timedelta(days=6)
start_date = recent_day - delta_week
mask_week = (df['sample_date'] >= start_date) & (df['sample_date'] <= recent_day)
df_week = df[mask_week]

# Mask biweekly
recent_day = max(df["run_date"]) 
delta_week = timedelta(days=13) 
start_date = recent_day - delta_week
mask_biweek = (df['sample_date'] >= start_date) & (df['sample_date'] <= recent_day)
df_biweek = df[mask_biweek]

# Mask monthly
recent_day = max(df["run_date"]) 
delta_week = timedelta(days=31) 
start_date = recent_day - delta_week
mask_month = (df['sample_date'] >= start_date) & (df['sample_date'] <= recent_day)
df_month = df[mask_month]
mask_monthN2 = (N2_conc['sample_date'] >= start_date) & (N2_conc['sample_date'] <= recent_day)
N2_conc = N2_conc[mask_monthN2]

if status_data == 'all data':
    recent_day = max(df["run_date"]) 
    delta_week = timedelta(days=365) 
    start_date = recent_day - delta_week
    print(start_date)
    mask_all = (df['run_date'] >= start_date) & (df['run_date'] <= recent_day)
    df_all = df[mask_all]

df_all=df_all.dropna(subset=['PCR_ID'])
# %% Create all plots for website

#! Plot 1 Trend of SARS N1 N2
path = '/Volumes/BAG_Data/'
# Save all .xlsx files paths and modification time into paths
paths = [(p.stat().st_mtime, p) for p in Path(path).iterdir() if p.suffix == ".csv"]
# Sort them by the modification time
paths = sorted(paths, key=lambda x: x[0], reverse=True)
# Get the last modified file and unlock
last = str(paths[0][1])
print(last)

SARS_conc = pd.DataFrame()
SARS_conc = pd.read_csv(last, sep = ';')
SARS_conc['date_collection'] = pd.to_datetime(SARS_conc['date_collection'])
trend_df = pd.DataFrame()
delt = (max(SARS_conc['date_collection']) - timedelta(days = 95))
trend_df = SARS_conc[(SARS_conc['date_collection']>delt)]
trend = figure(title="SARS-N1/N2 Concentration [gc/L]", x_axis_label="Sample Date", y_axis_label="SARS-N1/N2[gc/L]",
                            x_axis_type = 'datetime')

# Process each treatment plants data frame to have a 7day average, mean, and a linear regression of
# the last 7 days to get if the data is trending down or up

# Lugano
df_05_month = trend_df[(trend_df.ARA_ID == 515100)]
df_05_month['7day'] = df_05_month.conc_cov_target_N1_N2_E_M_S.rolling(7).mean()
N2_05 = N2_conc[(N2_conc.wwtp == 'CDA Lugano')]
N2_05['7day'] = N2_05['SARS-N2_(gc/mLWW)'].rolling(7).mean()
N2_05 = N2_05.groupby(['sample_date']).mean()
N2_05 = N2_05.reset_index()
slope5, intercept5, r_value5, p_value5, std_err5 = linregress(df_05_month['date_collection'].values.astype("float64")[-7:], df_05_month['7day'].values.astype("float64")[-7:])

# Zurich
df_10_month = trend_df[(trend_df.ARA_ID == 26101)]
df_10_month['7day'] = df_10_month.conc_cov_target_N1_N2_E_M_S.rolling(7).mean()
N2_10 = N2_conc[(N2_conc.wwtp == 'ARA Werdhölzli')]
N2_10['7day'] = N2_10['SARS-N2_(gc/mLWW)'].rolling(7).mean()
N2_10 = N2_10.groupby(['sample_date']).mean()
N2_10 = N2_10.reset_index()
slope10, intercept10, r_value10, p_value10, std_err10 = linregress(df_10_month['date_collection'].values.astype("float64")[-7:], df_10_month['7day'].values.astype("float64")[-7:])

# Geneva
df_16_month = trend_df[(trend_df.ARA_ID == 664301)]
df_16_month['7day'] = df_16_month.conc_cov_target_N1_N2_E_M_S.rolling(7).mean()
N2_16 = N2_conc[(N2_conc.wwtp == 'STEP Aire')]
N2_16['7day'] = N2_16['SARS-N2_(gc/mLWW)'].rolling(7).mean()
N2_16 = N2_16.groupby(['sample_date']).mean()
N2_16 = N2_16.reset_index()
slope16, intercept16, r_value16, p_value16, std_err16 = linregress(df_16_month['date_collection'].values.astype("float64")[-7:], df_16_month['7day'].values.astype("float64")[-7:])

# Chur
df_17_month = trend_df[(trend_df.ARA_ID == 390101)]
df_17_month['7day'] = df_17_month.conc_cov_target_N1_N2_E_M_S.rolling(7).mean()
N2_17 = N2_conc[(N2_conc.wwtp == 'ARA Chur')]
N2_17['7day'] = N2_17['SARS-N2_(gc/mLWW)'].rolling(7).mean()
N2_17 = N2_17.groupby(['sample_date']).mean()
N2_17 = N2_17.reset_index()
slope17, intercept17, r_value17, p_value17, std_err17 = linregress(df_17_month['date_collection'].values.astype("float64")[-7:], df_17_month['7day'].values.astype("float64")[-7:])

# Altenrhein
df_19_month = trend_df[(trend_df.ARA_ID == 323700)]
df_19_month['7day'] = df_19_month.conc_cov_target_N1_N2_E_M_S.rolling(7).mean()
N2_19 = N2_conc[(N2_conc.wwtp == 'ARA Altenrhein')]
N2_19['7day'] = N2_19['SARS-N2_(gc/mLWW)'].rolling(7).mean()
N2_19 = N2_19.groupby(['sample_date']).mean()
N2_19 = N2_19.reset_index()
slope19, intercept19, r_value19, p_value19, std_err19 = linregress(df_19_month['date_collection'].values.astype("float64")[-7:], df_19_month['7day'].values.astype("float64")[-7:])

# Sensetal
df_25_month = trend_df[(trend_df.ARA_ID == 66700)]
df_25_month['7day'] = df_25_month.conc_cov_target_N1_N2_E_M_S.rolling(7).mean()
N2_25 = N2_conc[(N2_conc.wwtp == 'ARA Sensetal')]
N2_25['7day'] = N2_25['SARS-N2_(gc/mLWW)'].rolling(7).mean()
N2_25 = N2_25.groupby(['sample_date']).mean()
N2_25 = N2_25.reset_index()
slope25, intercept25, r_value25, p_value25, std_err25 = linregress(df_25_month['date_collection'].values.astype("float64")[-7:], df_25_month['7day'].values.astype("float64")[-7:])

# Add multiple renderers for each treatment plant
trend.scatter(df_05_month['date_collection'], df_05_month['conc_cov_target_N1_N2_E_M_S'], legend_label="Lugano N1", color="red", line_width=3)
trend.scatter(df_10_month['date_collection'], df_10_month['conc_cov_target_N1_N2_E_M_S'], legend_label="Zurich N1", color="blue", line_width=3)
trend.scatter(df_16_month['date_collection'], df_16_month['conc_cov_target_N1_N2_E_M_S'], legend_label="Geneva N1", color="green", line_width=3)
trend.scatter(df_17_month['date_collection'], df_17_month['conc_cov_target_N1_N2_E_M_S'], legend_label="Chur N1", color="black", line_width=3)
trend.scatter(df_19_month['date_collection'], df_19_month['conc_cov_target_N1_N2_E_M_S'], legend_label="Altenrhein N1", color="grey", line_width=3)
trend.scatter(df_25_month['date_collection'], df_25_month['conc_cov_target_N1_N2_E_M_S'], legend_label="Laupen N1", color="darkgoldenrod", line_width=3)

trend.line(df_05_month['date_collection'], df_05_month['7day'], legend_label="Lugano N1", color="red", line_width=3)
trend.line(df_10_month['date_collection'], df_10_month['7day'], legend_label="Zurich N1", color="blue", line_width=3)
trend.line(df_16_month['date_collection'], df_16_month['7day'], legend_label="Geneva N1", color="green", line_width=3)
trend.line(df_17_month['date_collection'], df_17_month['7day'], legend_label="Chur N1", color="black", line_width=3)
trend.line(df_19_month['date_collection'], df_19_month['7day'], legend_label="Altenrhein N1", color="grey", line_width=3)
trend.line(df_25_month['date_collection'], df_25_month['7day'], legend_label="Laupen N1", color="darkgoldenrod", line_width=3)

trend.scatter(N2_05['sample_date'],N2_05['SARS-N2_(gc/mLWW)']*1000, legend_label="Lugano N2", color = "lightcoral", line_width = 3)
trend.scatter(N2_10['sample_date'],N2_10['SARS-N2_(gc/mLWW)']*1000, legend_label="Zurich N2", color = "lightblue", line_width = 3)
trend.scatter(N2_16['sample_date'],N2_16['SARS-N2_(gc/mLWW)']*1000, legend_label="Geneva N2", color = "lightgreen", line_width = 3)
trend.scatter(N2_17['sample_date'],N2_17['SARS-N2_(gc/mLWW)']*1000, legend_label="Chur N2", color = "darkslategrey", line_width = 3)
trend.scatter(N2_19['sample_date'],N2_19['SARS-N2_(gc/mLWW)']*1000, legend_label="Alterhein N2", color = "lightgrey", line_width = 3)
trend.scatter(N2_19['sample_date'],N2_19['SARS-N2_(gc/mLWW)']*1000, legend_label="Laupen N2", color = "goldenrod", line_width = 3)

trend.line(N2_05['sample_date'],N2_05['7day']*1000, legend_label="Lugano N2", color = "lightcoral", line_width = 3)
trend.line(N2_10['sample_date'],N2_10['7day']*1000, legend_label="Zurich N2", color = "lightblue", line_width = 3)
trend.line(N2_16['sample_date'],N2_16['7day']*1000, legend_label="Geneva N2", color = "lightgreen", line_width = 3)
trend.line(N2_17['sample_date'],N2_17['7day']*1000, legend_label="Chur N2", color = "darkslategrey", line_width = 3)
trend.line(N2_19['sample_date'],N2_19['7day']*1000, legend_label="Alterhein N2", color = "lightgrey", line_width = 3)
trend.line(N2_19['sample_date'],N2_19['7day']*1000, legend_label="Laupen N2", color = "goldenrod", line_width = 3)

trend.xgrid.grid_line_color = None
trend.ygrid.grid_line_color = None
trend.xgrid.grid_line_color = None
trend.ygrid.grid_line_color = None
trend.title.text_font_size = '17pt'
trend.xaxis.axis_label_text_font_size = "12pt"
trend.yaxis.axis_label_text_font_size = "12pt"
trend.width = 1200
trend.height = 400
trend.legend.location = 'top_left'
trend.y_range = Range1d(0, 2.5e6)
trend.xaxis[0].ticker.desired_num_ticks = 30

#! Plot 2 Monthly Fail Rate
# create a new plot with a title and axis labels
line_fail_month = figure(title="Failure Rate by Day", x_axis_label="Run Date", y_axis_label="Failure Rate", 
                    x_axis_type='datetime')

df_Q_month = df_month[(df_month.PCR_ID == 'Q')]
df_U_month = df_month[(df_month.PCR_ID == 'U')]
df_V_month = df_month[(df_month.PCR_ID == 'V')]
df_W_month = df_month[(df_month.PCR_ID == 'W')]
df_Y_month = df_month[(df_month.PCR_ID == 'Y')]
df_Z_month = df_month[(df_month.PCR_ID == 'Z')]

# add multiple renderers
line_fail_month.scatter(df_Q_month['run_date'], df_Q_month['failure_rate'], legend_label="Quinella", color="orchid", line_width=3)
line_fail_month.scatter(df_U_month['run_date'], df_U_month['failure_rate'], legend_label="Ulysses", color="cyan", line_width=3)
line_fail_month.scatter(df_V_month['run_date'], df_V_month['failure_rate'], legend_label="Vreni", color="mediumslateblue", line_width=3)
line_fail_month.scatter(df_W_month['run_date'], df_W_month['failure_rate'], legend_label="Winston", color="dodgerblue", line_width=3)
line_fail_month.scatter(df_Y_month['run_date'], df_Y_month['failure_rate'], legend_label="Yodok", color="sandybrown", line_width=3)
line_fail_month.scatter(df_Z_month['run_date'], df_Z_month['failure_rate'], legend_label="Zelda", color="olivedrab", line_width=3)
line_fail_month.xgrid.grid_line_color = None
line_fail_month.ygrid.grid_line_color = None
line_fail_month.xgrid.grid_line_color = None
line_fail_month.ygrid.grid_line_color = None
line_fail_month.title.text_font_size = '17pt'
line_fail_month.xaxis.axis_label_text_font_size = "12pt"
line_fail_month.yaxis.axis_label_text_font_size = "12pt"
line_fail_month.width = 400
line_fail_month.height = 400

line_recover_month = figure(title="MHV Recovery by Day", x_axis_label="Sample Date", y_axis_label="MHV Recovery",
                            x_axis_type = 'datetime')

df_month.sort_values(by=['sample_date'], inplace=True)

df_05_month = df_month[(df_month.wwtp == 'CDA Lugano') & (df_month.MHV_PercRecovery > 0.9) & (df_month.TotalDroplets > 15000)]
df_10_month = df_month[(df_month.wwtp == 'ARA Werdhölzli') & (df_month.MHV_PercRecovery > 0.9) & (df_month.TotalDroplets > 15000)]
df_16_month = df_month[(df_month.wwtp == 'STEP Aire') & (df_month.MHV_PercRecovery > 0.9) & (df_month.TotalDroplets > 15000)]
df_17_month = df_month[(df_month.wwtp == 'ARA Chur') & (df_month.MHV_PercRecovery > 0.9) & (df_month.TotalDroplets > 15000)]
df_19_month = df_month[(df_month.wwtp == 'ARA Altenrhein') & (df_month.MHV_PercRecovery > 0.9) & (df_month.TotalDroplets > 15000)]
df_25_month = df_month[(df_month.wwtp == 'ARA Sensetal') & (df_month.MHV_PercRecovery > 0.9) & (df_month.TotalDroplets > 15000)]

df_05_month['MHV7'] = df_05_month.MHV_PercRecovery.rolling(3).mean()
df_10_month['MHV7'] = df_10_month.MHV_PercRecovery.rolling(3).mean()
df_16_month['MHV7'] = df_16_month.MHV_PercRecovery.rolling(3).mean()
df_17_month['MHV7'] = df_17_month.MHV_PercRecovery.rolling(3).mean()
df_19_month['MHV7'] = df_19_month.MHV_PercRecovery.rolling(3).mean()
df_25_month['MHV7'] = df_25_month.MHV_PercRecovery.rolling(3).mean()

# add multiple renderers
line_recover_month.scatter(df_05_month['sample_date'], df_05_month['MHV_PercRecovery'], legend_label="Lugano", color="red", line_width=4)
line_recover_month.scatter(df_10_month['sample_date'], df_10_month['MHV_PercRecovery'], legend_label="Zurich", color="blue", line_width=4)
line_recover_month.scatter(df_16_month['sample_date'], df_16_month['MHV_PercRecovery'], legend_label="Geneva", color="green", line_width=4)
line_recover_month.scatter(df_17_month['sample_date'], df_17_month['MHV_PercRecovery'], legend_label="Chur", color="black", line_width=4)
line_recover_month.scatter(df_19_month['sample_date'], df_19_month['MHV_PercRecovery'], legend_label="Altenrhein", color="grey", line_width=4)
line_recover_month.scatter(df_25_month['sample_date'], df_25_month['MHV_PercRecovery'], legend_label="Laupen", color="pink", line_width=4)

line_recover_month.line(df_05_month['sample_date'], df_05_month['MHV7'], legend_label="Lugano", color="red", line_width=4)
line_recover_month.line(df_10_month['sample_date'], df_10_month['MHV7'], legend_label="Zurich", color="blue", line_width=4)
line_recover_month.line(df_16_month['sample_date'], df_16_month['MHV7'], legend_label="Geneva", color="green", line_width=4)
line_recover_month.line(df_17_month['sample_date'], df_17_month['MHV7'], legend_label="Chur", color="black", line_width=4)
line_recover_month.line(df_19_month['sample_date'], df_19_month['MHV7'], legend_label="Altenrhein", color="grey", line_width=4)
line_recover_month.line(df_25_month['sample_date'], df_25_month['MHV7'], legend_label="Laupen", color="pink", line_width=4)

line_recover_month.xgrid.grid_line_color = None
line_recover_month.ygrid.grid_line_color = None
line_recover_month.xgrid.grid_line_color = None
line_recover_month.ygrid.grid_line_color = None
line_recover_month.title.text_font_size = '17pt'
line_recover_month.xaxis.axis_label_text_font_size = "12pt"
line_recover_month.yaxis.axis_label_text_font_size = "12pt"
line_recover_month.width = 400
line_recover_month.height = 400
line_recover_month.legend.location = 'top_left'
line_recover_month.y_range = Range1d(0, 60)

#! Plot 4
line_inhib_week = figure(title="Inhibition by Day", x_axis_label="Sample Date", y_axis_label="Inhibition",
                            x_axis_type = 'datetime')

df_05_week = df_biweek[(df_biweek.wwtp == 'CDA Lugano') & (df_biweek.spiked == False) & (df_month.TotalDroplets > 15000)]
df_10_week = df_biweek[(df_biweek.wwtp == 'ARA Werdhölzli') & (df_biweek.spiked == False) & (df_month.TotalDroplets > 15000)]
df_16_week = df_biweek[(df_biweek.wwtp == 'STEP Aire') & (df_biweek.spiked == False) & (df_month.TotalDroplets > 15000)]
df_17_week = df_biweek[(df_biweek.wwtp == 'ARA Chur') & (df_biweek.spiked == False) & (df_month.TotalDroplets > 15000)]
df_19_week = df_biweek[(df_biweek.wwtp == 'ARA Altenrhein') & (df_biweek.spiked == False) & (df_month.TotalDroplets > 15000)]
df_25_week = df_biweek[(df_biweek.wwtp == 'ARA Sensetal') & (df_biweek.spiked == False) & (df_month.TotalDroplets > 15000)]

# add multiple renderers
line_inhib_week.scatter(df_05_week['sample_date'], df_05_week['SARS-N1_InhibitionQuant'], legend_label="Lugano", color="red", line_width=4)
line_inhib_week.scatter(df_10_week['sample_date'], df_10_week['SARS-N1_InhibitionQuant'], legend_label="Zurich", color="blue", line_width=4)
line_inhib_week.scatter(df_16_week['sample_date'], df_16_week['SARS-N1_InhibitionQuant'], legend_label="Geneva", color="green", line_width=4)
line_inhib_week.scatter(df_17_week['sample_date'], df_17_week['SARS-N1_InhibitionQuant'], legend_label="Chur", color="black", line_width=4)
line_inhib_week.scatter(df_19_week['sample_date'], df_19_week['SARS-N1_InhibitionQuant'], legend_label="Altenrhein", color="grey", line_width=4)
line_inhib_week.scatter(df_25_week['sample_date'], df_25_week['SARS-N1_InhibitionQuant'], legend_label="Laupen", color="pink", line_width=4)

line_inhib_week.xgrid.grid_line_color = None
line_inhib_week.ygrid.grid_line_color = None
line_inhib_week.xgrid.grid_line_color = None
line_inhib_week.ygrid.grid_line_color = None
line_inhib_week.title.text_font_size = '17pt'
line_inhib_week.xaxis.axis_label_text_font_size = "12pt"
line_inhib_week.yaxis.axis_label_text_font_size = "12pt"
line_inhib_week.width = 400
line_inhib_week.height = 400
line_inhib_week.legend.location = 'bottom_left'
line_inhib_week.y_range = Range1d(0, 1.2)
inhib_best = Span(location=0.6,
                              dimension='width', line_color='red', line_width=3)
line_inhib_week.add_layout(inhib_best)


trend.legend.click_policy="hide"
line_inhib_week.legend.click_policy="hide"
line_recover_month.legend.click_policy="hide"
line_fail_month.legend.click_policy="hide"

inhib_best = Span(location=0.6,
                              dimension='height', line_color='red', line_width=3)
line_inhib_week.add_layout(inhib_best)

# Checking which samples are here
today = date.today()
print('day Name:', today.strftime('%A'))
td = today.strftime('%A')

if td == 'Friday' or td == 'Saturday' or td == 'Sunday' or td == 'Monday' or td == 'Tuesday':  
    offset = (today.weekday() - 1 - TUESDAY) % 7
    print (offset)
    last_tuesday = today - timedelta(days=offset+1) 
    print(last_tuesday)
    updaterecent = 'The most recent data should be up to: ' + last_tuesday.strftime("%Y-%m-%d") + ' \n'
    print(updaterecent)
    
    framing_start = last_tuesday - timedelta(days=6)
    print(framing_start)
    weekdates = pd.date_range(start=framing_start, end=last_tuesday)
    print(weekdates)

    # Lugano Check
    valid_data_05 = df_05_week[(df_05_week['sample_date']<=max(weekdates)) & (df_05_week['sample_type'] == 'ww') & (df_05_week.TotalDroplets > 15000)]
    dates05 = np.unique(valid_data_05['sample_date'])
    missing = ''
    for x in weekdates:
        vect = x == dates05
        if not any(vect) == True:
            missing = missing + '\n' + '05_'+ x.strftime("%Y-%m-%d")
    missing = missing + '<br><br>'

    # Zurich Check
    valid_data_10 = df_10_week[(df_10_week['sample_date']<=max(weekdates)) & (df_10_week['sample_type'] == 'ww') & (df_10_week.TotalDroplets > 15000)]
    dates10 = np.unique(valid_data_10['sample_date'])
    for x in weekdates:
        vect = x == dates10
        if not any(vect) == True:
            missing = missing + '\n' + '10_'+ x.strftime("%Y-%m-%d")
    missing = missing + '<br><br>'   

    # Geneva Check
    valid_data_16 = df_16_week[(df_16_week['sample_date']<=max(weekdates)) & (df_16_week['sample_type'] == 'ww') & (df_16_week.TotalDroplets > 15000)]
    dates16 = np.unique(valid_data_16['sample_date'])
    for x in weekdates:
        vect = x == dates16
        if not any(vect) == True:
            missing = missing + '\n' + '16_'+ x.strftime("%Y-%m-%d")
    missing = missing + '<br><br>'   

    # Chur Check
    valid_data_17 = df_17_week[(df_17_week['sample_date']<=max(weekdates)) & (df_17_week['sample_type'] == 'ww') & (df_17_week.TotalDroplets > 15000)]
    dates17 = np.unique(valid_data_17['sample_date'])
    for x in weekdates:
        vect = x == dates17
        if not any(vect) == True:
            missing = missing + '\n' + '17_'+ x.strftime("%Y-%m-%d")
    missing = missing + '<br><br>'   

    # Altenrhein Check
    valid_data_19 = df_19_week[(df_19_week['sample_date']<=max(weekdates)) & (df_19_week['sample_type'] == 'ww') & (df_19_week.TotalDroplets > 15000)]
    dates19 = np.unique(valid_data_19['sample_date'])
    for x in weekdates:
        vect = x == dates19
        if not any(vect) == True:
            missing = missing + '\n' + '19_'+ x.strftime("%Y-%m-%d")
    missing = missing + '<br><br>'  

    # Laupen Check
    valid_data_25 = df_25_week[(df_25_week['sample_date']<=max(weekdates)) & (df_25_week['sample_type'] == 'ww') & (df_25_week.TotalDroplets > 15000)]
    dates25 = np.unique(valid_data_25['sample_date'])
    for x in weekdates:
        vect = x == dates25
        if not any(vect) == True:
            missing = missing + '\n' + '25_'+ x.strftime("%Y-%m-%d")

    print(missing)

elif td == 'Wednesday' : 
    offset = (today.weekday() - 1 - SUNDAY) % 7
    print (offset)
    last_sunday = today - timedelta(days=offset+1) 
    print(last_sunday)
    updaterecent = 'The most recent data should be up to: ' + last_sunday.strftime("%Y-%m-%d") + ' \n'
    print(updaterecent)
    
    framing_start = last_sunday - timedelta(days=4)
    print(framing_start)
    weekdates = pd.date_range(start=framing_start, end=last_sunday)
    print(weekdates)

        # Lugano Check
    valid_data_05 = df_05_week[(df_05_week['sample_date']<=max(weekdates)) & (df_05_week['sample_type'] == 'ww') & (df_05_week.TotalDroplets > 15000)]
    dates05 = np.unique(valid_data_05['sample_date'])
    missing = ''
    for x in weekdates:
        vect = x == dates05
        if not any(vect) == True:
            missing = missing + '\n' + '05_'+ x.strftime("%Y-%m-%d")
    missing = missing + '<br><br>'

    # Zurich Check
    valid_data_10 = df_10_week[(df_10_week['sample_date']<=max(weekdates)) & (df_10_week['sample_type'] == 'ww') & (df_10_week.TotalDroplets > 15000)]
    dates10 = np.unique(valid_data_10['sample_date'])
    for x in weekdates:
        vect = x == dates10
        if not any(vect) == True:
            missing = missing + '\n' + '10_'+ x.strftime("%Y-%m-%d")
    missing = missing + '<br><br>'   

    # Geneva Check
    valid_data_16 = df_16_week[(df_16_week['sample_date']<=max(weekdates)) & (df_16_week['sample_type'] == 'ww') & (df_16_week.TotalDroplets > 15000)]
    dates16 = np.unique(valid_data_16['sample_date'])
    for x in weekdates:
        vect = x == dates16
        if not any(vect) == True:
            missing = missing + '\n' + '16_'+ x.strftime("%Y-%m-%d")

elif today.strftime('%A') == 'Thursday' :
    offset = (today.weekday() - 1 - SUNDAY) % 7
    print (offset)
    last_sunday = today - timedelta(days=offset+1) 
    print(last_sunday)
    updaterecent = 'The most recent data should be up to: ' + last_sunday.strftime("%Y-%m-%d") + ' \n'
    print(updaterecent)
    
    framing_start = last_sunday - timedelta(days=4)
    print(framing_start)
    weekdates = pd.date_range(start=framing_start, end=last_sunday)
    print(weekdates)

    # Lugano Check
    valid_data_05 = df_05_week[(df_05_week['sample_date']<=max(weekdates)) & (df_05_week['sample_type'] == 'ww') & (df_05_week.TotalDroplets > 15000)]
    dates05 = np.unique(valid_data_05['sample_date'])
    missing = ''
    for x in weekdates:
        vect = x == dates05
        if not any(vect) == True:
            missing = missing + '\n' + '05_'+ x.strftime("%Y-%m-%d")
    missing = missing + '<br><br>'

    # Zurich Check
    valid_data_10 = df_10_week[(df_10_week['sample_date']<=max(weekdates)) & (df_10_week['sample_type'] == 'ww') & (df_10_week.TotalDroplets > 15000)]
    dates10 = np.unique(valid_data_10['sample_date'])
    for x in weekdates:
        vect = x == dates10
        if not any(vect) == True:
            missing = missing + '\n' + '10_'+ x.strftime("%Y-%m-%d")
    missing = missing + '<br><br>'   

    # Geneva Check
    valid_data_16 = df_16_week[(df_16_week['sample_date']<=max(weekdates)) & (df_16_week['sample_type'] == 'ww') & (df_16_week.TotalDroplets > 15000)]
    dates16 = np.unique(valid_data_16['sample_date'])
    for x in weekdates:
        vect = x == dates16
        if not any(vect) == True:
            missing = missing + '\n' + '16_'+ x.strftime("%Y-%m-%d")
    missing = missing + '<br><br>'   

    # Chur Check
    valid_data_17 = df_17_week[(df_17_week['sample_date']<=max(weekdates)) & (df_17_week['sample_type'] == 'ww') & (df_17_week.TotalDroplets > 15000)]
    dates17 = np.unique(valid_data_17['sample_date'])
    for x in weekdates:
        vect = x == dates17
        if not any(vect) == True:
            missing = missing + '\n' + '17_'+ x.strftime("%Y-%m-%d")
    missing = missing + '<br><br>'   

    # Altenrhein Check
    valid_data_19 = df_19_week[(df_19_week['sample_date']<=max(weekdates)) & (df_19_week['sample_type'] == 'ww') & (df_19_week.TotalDroplets > 15000)]
    dates19 = np.unique(valid_data_19['sample_date'])
    for x in weekdates:
        vect = x == dates19
        if not any(vect) == True:
            missing = missing + '\n' + '19_'+ x.strftime("%Y-%m-%d")
    missing = missing + '<br><br>'  

    # Laupen Check
    valid_data_25 = df_25_week[(df_25_week['sample_date']<=max(weekdates)) & (df_25_week['sample_type'] == 'ww') & (df_25_week.TotalDroplets > 15000)]
    dates25 = np.unique(valid_data_25['sample_date'])
    for x in weekdates:
        vect = x == dates25
        if not any(vect) == True:
            missing = missing + '\n' + '25_'+ x.strftime("%Y-%m-%d")

if missing == '<br><br><br><br><br><br><br><br><br><br>':
    missing = 'All wastewater data is up to date <br><br>'

if 'valid_data_05' in locals():
     dil_05 = valid_data_05['dilution']
     dil_05 = np.unique(dil_05)
     dil_05 = str(dil_05[0])
if 'valid_data_10' in locals():
     dil_10 = valid_data_10['dilution']
     dil_10 = np.unique(dil_10)
     dil_10 = str(dil_10[0])
if 'valid_data_16' in locals():
     dil_16 = valid_data_16['dilution']
     dil_16 = np.unique(dil_16)
     dil_16 = str(dil_16[0])
if 'valid_data_17' in locals():
     dil_17 = valid_data_17['dilution']
     dil_17 = np.unique(dil_17)
     dil_17 = str(dil_17[0])
if 'valid_data_19' in locals():
     dil_19 = valid_data_19['dilution']
     dil_19 = np.unique(dil_19)
     dil_19 = str(dil_19[0])
if 'valid_data_25' in locals():
     dil_25 = valid_data_25['dilution']
     dil_25 = np.unique(dil_25)
     dil_25 = str(dil_25[0])

# Find Missing Case and Flow Data
case_flow = df_case_flow['sample_name']
mask_case = (df_case_flow['new_cases [1/(d*100000 capita)]'].isnull())
mask_flow = (df_case_flow['flow [m^3/d]'].isnull())
cases_miss = case_flow[mask_case]
cases_miss = cases_miss.sort_values()
flows_miss = case_flow[mask_flow]
flows_miss = flows_miss.sort_values()
cases_miss = cases_miss.to_string(index = False)
flows_miss = flows_miss.to_string(index = False)

if (flows_miss == 'Series([], )') == True:
    flows_miss = 'All flow data is up to date'
if (cases_miss == 'Series([], )') == True:
    cases_miss = 'All case data is up to date'

# Find LOD and LOQ
mask_lod = (df_loq_lod['quantification_flag [{Q: >LOQ,D: >LOD,N: <LOD}]']=='D')
mask_loq = (df_loq_lod['quantification_flag [{Q: >LOQ,D: >LOD,N: <LOD}]']=='Q')
mask_ND = (df_loq_lod['quantification_flag [{Q: >LOQ,D: >LOD,N: <LOD}]']=='N')

samples_which = df_loq_lod['sample_name']
lod = samples_which[mask_lod]
loq = samples_which[mask_loq]
nd = samples_which[mask_ND]
lod = lod.to_string(index = False)
nd = nd.to_string(index = False)

if (lod == 'Series([], )') == True:
    lod = 'None below LOQ'
if (nd == 'Series([], )') == True:
    nd = 'None below LOD'

lod = '<b>Samples below LOQ are:</b> <br>' + lod
nd = '<b>Samples below LOD are:</b> <br>' + nd

if td == 'Wednesday' :
    dilution_text = ('<b>The current dilutions are:</b> <br><br> '
                        'Lugano: '+ dil_05 + 'x<br>'
                        'Zurich: ' + dil_10 + 'x<br>'
                        'Geneva: ' + dil_16 + 'x<br>')
else:
    dilution_text = ('<b>The current dilutions are:</b> <br><br> '
                    'Lugano: '+ dil_05 + 'x<br>'
                    'Zurich: ' + dil_10 + 'x<br>'
                    'Geneva: ' + dil_16 + 'x<br>'
                    'Chur: ' + dil_17 + 'x<br>'
                    'Altenrhein: ' + dil_19 + 'x<br>'
                    'Laupen: ' + dil_25 + 'x<br>')

df_week_controls = df_week[(df_week.sample_type == 'btc') |
                                (df_week.sample_type == 'fpcz') |
                                    (df_week.sample_type == 'ntc')]
df_week_controls = df_week_controls[(df_week.spiked == False)&
                                    df_week['SARS-N1_PositiveDroplets'] > 2 ]

if df_week_controls['SampleName'].empty == True :
    df_week_controls_div = 'All FPCZs, BTCs, and NTCs below 3 droplets'
else :
    df_week_controls_div = str(df_week_controls['SampleName'])

div = Div(text='<br><br>' + updaterecent +'<br><br>Page last updated: ' + str(today) + '<br><br>' + '<b>The samples that are missing this week are:</b> <br><br>' + missing
            + '<br><br>' + dilution_text + '<br><br>' + '<b>NTCs:</b><br>' + df_week_controls_div + '<br><br>'
            + '<b>Cases Missing:</b> <br>' + cases_miss + '<br><br>' + '<b>Flows Missing:</b> <br>' + flows_miss
            + '<br><br>' + lod + '<br><br>' + nd,
            width=200, height=100)

grid = layout([
    [trend, div],
    [line_inhib_week, line_recover_month, line_fail_month]
])

output_file(filename="layout.html", title="Covid Quality Control")
save(grid)

print(slope5*10e12,
slope10*10e12,
slope16*10e12,
slope17*10e12,
slope19*10e12,
slope25*10e12)

print(slope5*10e10,
slope10*10e10,
slope16*10e10,
slope17*10e10,
slope19*10e10,
slope25*10e10)

# %%
#df_month_test = df_all[(df_all.MHV_PercRecovery > 0.9) & (df_all.TotalDroplets > 15000)]
#fig1 = figure()
#fig1.scatter(df_month_test['SARS-N1_InhibitionQuant'], df_month_test['MHV_PercRecovery'], legend_label="recovery/inhib", color="red", line_width=4)
#show(fig1)

df_05_week[(df_05_week['sample_date']>='2023-01-18') & (df_05_week['replicate'] == 'C')].to_csv('maxwell_lugano.csv')
df_05_week[(df_05_week['sample_date']>='2023-01-18') & (df_05_week['replicate'] == 'A')].to_csv('norm_lugano.csv')

df_10_week[(df_10_week['sample_date']>='2023-01-18') & (df_10_week['replicate'] == 'C')].to_csv('maxwell_zurich.csv')
df_10_week[(df_10_week['sample_date']>='2023-01-18') & (df_10_week['replicate'] == 'A')].to_csv('norm_zurich.csv')

df_16_week[(df_16_week['sample_date']>='2023-01-18') & (df_16_week['replicate'] == 'C')].to_csv('maxwell_geneva.csv')
df_16_week[(df_16_week['sample_date']>='2023-01-18') & (df_16_week['replicate'] == 'A')].to_csv('norm_geneva.csv')

df_17_week[(df_17_week['sample_date']>='2023-01-18') & (df_17_week['replicate'] == 'C')].to_csv('maxwell_chur.csv')
df_17_week[(df_17_week['sample_date']>='2023-01-18') & (df_17_week['replicate'] == 'A')].to_csv('norm_chur.csv')

df_19_week[(df_19_week['sample_date']>='2023-01-18') & (df_19_week['replicate'] == 'C')].to_csv('maxwell_altenrhein.csv')
df_19_week[(df_19_week['sample_date']>='2023-01-18') & (df_19_week['replicate'] == 'A')].to_csv('norm_altenrhein.csv')

df_25_week[(df_25_week['sample_date']>='2023-01-18') & (df_25_week['replicate'] == 'C')].to_csv('maxwell_laupen.csv')
df_25_week[(df_25_week['sample_date']>='2023-01-18') & (df_25_week['replicate'] == 'A')].to_csv('norm_laupen.csv')

# %%
