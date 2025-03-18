
import re
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

"""### Preprocessing filters"""


class ColumnName:
    def __init__(self, value):
        self.value = value

def concat_cell_names_indigo(df):
    """
    For planned indigo cells:
    CELL = CELL + _P + PLAN_VERSION

    Parameters
    ----------
    indigo_df : pd.DataFrame
        A DataFrame containing data from Indigo database

    Returns
    -------
    indigo_df : pd.DataFrame
        indigo DataFrame with CELL column values changed
    """
    indigo_df = df.copy()

    # Create a boolean mask where the plan status is 'P'
    mask = indigo_df["PLAN_STATUS"] == "P"

    # Apply the transformation only on the rows where mask is True
    indigo_df.loc[mask, "CELL"] = indigo_df.loc[mask, "CELL"] + "_P" + indigo_df.loc[mask, "PLAN_VERSION"].astype(int).astype(str)

    #Apply condition on NISV to change 1900-01-01 with string
    indigo_df['NISV_ERHALTEN_KONTR_VERLINKT'] =indigo_df['NISV_ERHALTEN_KONTR_VERLINKT'].fillna("")
    indigo_df['NISV_ERHALTEN_KONTR_VERLINKT'] =indigo_df['NISV_ERHALTEN_KONTR_VERLINKT'].astype(str)
    indigo_df['NISV_ERHALTEN_KONTR_VERLINKT']=indigo_df.apply(lambda x: "" if x['NISV_ERHALTEN_KONTR_VERLINKT']=="1900-01-01 00:00:00" else x['NISV_ERHALTEN_KONTR_VERLINKT'], axis=1)
    indigo_df['NISV_ERHALTEN_KONTR_VERLINKT'] =indigo_df['NISV_ERHALTEN_KONTR_VERLINKT'].astype(str)

    indigo_df['NISV_ERHALTEN_KONTR_VERLINKT']=indigo_df['NISV_ERHALTEN_KONTR_VERLINKT'].str.split(':').str[0].str.split(' ').str[0]
    #indigo_df['NISV_ERHALTEN_KONTR_VERLINKT'].replace('nan', '', inplace=True)
    indigo_df['NISV_ERHALTEN_KONTR_VERLINKT']=pd.to_datetime(indigo_df['NISV_ERHALTEN_KONTR_VERLINKT'],format="%d.%m.%Y")
    indigo_df.loc[indigo_df['NISV_ERHALTEN_KONTR_VERLINKT'] == "01-01-1900", 'NISV_ERHALTEN_KONTR_VERLINKT'] = ""

    return indigo_df


def get_indigo_planned_cells(indigo_df):
    """
    TODO
    """
    df = indigo_df.copy()
    considered_operative_mask = (df['PLAN_STATUS'] == 'O') & ((df['OPERATIVE_CELL_STATUS'] == 'OMC_del') | (df['OPERATIVE_CELL_STATUS'] == 'none'))
    considered_planned_mask = df['PLAN_STATUS'] == 'P'

    return df[considered_operative_mask | considered_planned_mask]


def merge_indigo_with_asset_5g(asset_5g_df, indigo_df):
    """
    TODO
    """
    pass


def get_asset_planned_cells(asset_df, indigo_df):
    """
    TODO
    """
    indigo_planned = get_indigo_planned_cells(indigo_df)

    indigo_copy = indigo_planned.copy()
    indigo_copy['Identity'] = indigo_copy['CELL']
    indigo_copy["Identity"] = indigo_copy["Identity"] + "_P" + indigo_copy["PLAN_VERSION"].astype(int).astype(str)

    indigo_copy = indigo_copy.drop_duplicates(subset='Identity', keep='first')
    merged_df = pd.merge(asset_df,
              indigo_copy[['Identity']],
              how='left',
              on='Identity',
                indicator=True)

    merged_df.index = asset_df.index

    # take only-asset cells and concacanate with merhge-both
    merged_df = merged_df[(merged_df['_merge'] == 'both')]

    indigo_full = indigo_df.copy()
    indigo_full['Identity'] = np.where(indigo_full['PLAN_STATUS'] == 'O',
                                   indigo_full['CELL'],
                                   indigo_full['CELL'] + "_P" + indigo_full['PLAN_VERSION'].astype(int).astype(str))

    indigo_full = indigo_full.drop_duplicates(subset=['Identity', ])
    only_asset = pd.merge(asset_df,
                          indigo_full,
                          how='left',
                          on='Identity',
                          indicator=True)
    only_asset.index = asset_df.index

    only_asset_planned = only_asset[only_asset['_merge'] == 'left_only']
    only_asset_planned = only_asset_planned[only_asset_planned['Identity'].str.len() > 6]

    combined_df = pd.concat([merged_df, only_asset_planned])
    return combined_df.drop(columns=['_merge', ])

def get_asset_outdoor_cells(asset_df):
    """
    TODO
    """
    return asset_df[asset_df['Cell_Coverage'].str.contains('Outdoor')]


def merge_asset_4g_with_indigo(asset_4g_df, indigo_df):
    indigo_df['Identity'] = indigo_df['CELL']
    indigo_df = indigo_df.drop_duplicates(subset='Identity', keep='first')

    df = pd.merge(asset_4g_df,
                  indigo_df[['Identity', 'NET_IDS', 'PREAMPLIFIER_CODE', 'PLAN_STATUS', 'OPERATIVE_CELL_STATUS', 'NISV_ERHALTEN_KONTR_VERLINKT','KANTON']],
                  on='Identity', how='left',
                  indicator=False)
    df.index = asset_4g_df.index

    return df


def merge_asset_5g_with_indigo(asset_5g_df, indigo_df):
    indigo_df['Identity'] = indigo_df['CELL']
    indigo_df = indigo_df.drop_duplicates(subset='Identity', keep='first')

    df = pd.merge(asset_5g_df,
                  indigo_df[['Identity', 'NET_IDS', 'PREAMPLIFIER_CODE', 'PLAN_STATUS', 'OPERATIVE_CELL_STATUS', 'NISV_ERHALTEN_KONTR_VERLINKT', 'DESIGNER',]],
                  on='Identity', how='left',
                  indicator=False)
    df.index = asset_5g_df.index

    return df

def merge_asset_3g_with_indigo(asset_3g_df, indigo_df):
    indigo_df['Identity'] = indigo_df['CELL']
    indigo_df = indigo_df.drop_duplicates(subset='Identity', keep='first')

    df = pd.merge(asset_3g_df,
                  indigo_df[['Identity', 'NET_IDS', 'PREAMPLIFIER_CODE', 'PLAN_STATUS', 'OPERATIVE_CELL_STATUS', 'NISV_ERHALTEN_KONTR_VERLINKT']],
                  on='Identity', how='left',
                  indicator=False)
    df.index = asset_3g_df.index

    return df


def is_active(antenna: str) -> bool:
    """
    Check if 5G antenna is active

    Parameters
    ----------
    antenna : str
        Antenna type name

    Return:
    _ : bool
    """
    antenna_pattern = re.compile(r'(AIR[a-zA-Z0-9]{4})|([a-zA-Z0-9]{4}-A)')
    try:
        return bool(antenna_pattern.search(antenna))
    except:
        # NaN values
        return False


def get_corrections(asset_df, corrections_list):
    """
    TODO
    """
    asset_copy = asset_df.copy()
    asset_copy['Corrections'] = [[] for _ in range(len(asset_df))]

    for idx, col, val in corrections_list:
        asset_copy.at[idx, col] = val
        current_value = asset_copy.at[idx, 'Corrections']
        try:
            current_value.append(col)
        except:
            break
        # asset_copy.at[idx, 'Corrections'] = current_value
        asset_copy.at[idx, 'Corrections'] = list(set(current_value))

    return asset_copy

def simple_correction_new(asset_df, col_variable, value, **conditions):
    df = asset_df.copy()

    # Extract column name dynamically from the variable
    col = col_variable.value  

    # Initialize a base condition that always evaluates to True
    condition = pd.Series(True, index=df.index)

    # Determine if new_value_or_func is a function or a direct value
    is_value_func = callable(value)

    # Process conditions
    for condition_col, condition_val in conditions.items():
        if isinstance(condition_val, tuple):
            if condition_val[0] == 'regex':
                # Regex condition
                regex_pattern = condition_val[1]
                condition &= df[condition_col].astype(str).str.contains(regex_pattern, regex=True, na=False)
            elif condition_val[0] == 'gt':
                # Greater than condition
                threshold = condition_val[1]
                condition &= df[condition_col] > threshold
            elif condition_val[0] == 'lt':
                # Less than condition
                threshold = condition_val[1]
                condition &= df[condition_col] < threshold
            elif condition_val[0] == 'range':
                # Range condition: values must be greater than value1 and less than value2
                lower_bound, upper_bound = condition_val[1]
                condition &= (df[condition_col] > lower_bound) & (df[condition_col] < upper_bound)
            elif condition_val[0] == 'neq':
                non_matching_value = condition_val[1]
                condition &= df[condition_col] != non_matching_value

        elif callable(condition_val):
            # This is a lambda or custom function condition
            condition &= df[condition_col].apply(condition_val)
        else:
            # Direct value comparison
            condition &= (df[condition_col] == condition_val)

    if isinstance(value, str):
        mask = condition & (df[col].str.strip().str.lower() != value.strip().lower())
    else:
        mask = condition & (df[col] != value)

    # Perform the correction and return the modified dataframe or indices
    if is_value_func:
        # If the new value is determined by a function, apply it row-wise
        df.loc[mask, col] = df.loc[mask].apply(lambda row: value(row), axis=1)
    else:
        # If a direct value is provided, simply use it
        df.loc[mask, col] = value

    # Get indices where the mask is True
    indices = df.index[mask].tolist()

    # Optionally, return indices of corrected rows
    return [(indx, col, df.at[indx, col] if col in df else None) for indx in indices]


def simple_correction(asset_df, col, value, **conditions):
    df = asset_df.copy()

    # Initialize a base condition that always evaluates to True
    condition = pd.Series(True, index=df.index)

    # Determine if new_value_or_func is a function or a direct value
    is_value_func = callable(value)

    # Process conditions
    for condition_col, condition_val in conditions.items():
        if isinstance(condition_val, tuple):
            if condition_val[0] == 'regex':
                # Regex condition
                regex_pattern = condition_val[1]
                condition &= df[condition_col].astype(str).str.contains(regex_pattern, regex=True, na=False)
            elif condition_val[0] == 'gt':
                # Greater than condition
                threshold = condition_val[1]
                condition &= df[condition_col] > threshold
            elif condition_val[0] == 'lt':
                # Less than condition
                threshold = condition_val[1]
                condition &= df[condition_col] < threshold
            elif condition_val[0] == 'range':
                # Range condition: values must be greater than value1 and less than value2
                lower_bound, upper_bound = condition_val[1]
                condition &= (df[condition_col] > lower_bound) & (df[condition_col] < upper_bound)
            elif condition_val[0] == 'neq':
                non_matching_value = condition_val[1]
                condition &= df[condition_col] != non_matching_value


        elif callable(condition_val):
            # This is a lambda or custom function condition
            condition &= df[condition_col].apply(condition_val)
        else:
            # Direct value comparison
            condition &= (df[condition_col] == condition_val)

    if isinstance(value, str):
        mask = condition & (df[col].str.strip().str.lower() != value.strip().lower())
    else:
        mask = condition & (df[col] != value)

    # Perform the correction and return the modified dataframe or indices
    if is_value_func:
        # If the new value is determined by a function, apply it row-wise
        df.loc[mask, col] = df.loc[mask].apply(lambda row: value(row), axis=1)
    else:
        # If a direct value is provided, simply use it
        df.loc[mask, col] = value

    # Get indices where the mask is True
    indices = df.index[mask].tolist()

    # Optionally, return indices of corrected rows
    return [(indx, col, df.at[indx, col] if col in df else None) for indx in indices]

def check_carrier_in_group(node_name, assigned_carrier, carrier_groups):
    """
    Check if the assigned carrier is in the carrier group of the given node name.

    Args:
    - node_name (str): The node name.
    - assigned_carrier (str): The assigned carrier to check.
    - carrier_groups (dict): A dictionary with node names as keys and lists of carriers as values.

    Returns:
    - bool: True if the assigned carrier is in the group, False otherwise.
    """
    if node_name in carrier_groups:
        return assigned_carrier in carrier_groups[node_name]
    return False

NODE_IDENTITY = 'Node IDENTITY'
CARRIER_ON_NODE_LEVEL = 'Carrier on node level'
def add_carrier_on_node_level(df, carrier_groups):
    """
    Add a 'Carrier on node level' column to the DataFrame based on the specified conditions.

    Args:
    - df (pd.DataFrame): The input DataFrame with columns 'NODE_NAME', 'Assigned Carrier', and 'Correction'.
    - carrier_groups (dict): A dictionary with node names as keys and lists of carriers as values.

    Returns:
    - pd.DataFrame: The DataFrame with the new 'Carrier on node level' column added.
    """
    def determine_carrier_on_node(row):
        if 'Assigned Carrier' in row['Corrections']:
            if not check_carrier_in_group(row[NODE_IDENTITY], row['Assigned Carrier'], carrier_groups):
                return row['Assigned Carrier']
        return "carrier available"

    df[CARRIER_ON_NODE_LEVEL] = df.apply(determine_carrier_on_node, axis=1)
    return df

# For MAX TX POWER:
def antenna_check(included=None, excluded=None):

    """
    Returns a lambda function that checks if 'included' substring is in the antenna value
    and 'excluded' substring is not, if provided.
    """
    if included and excluded is None:
        return lambda x: isinstance(x, str) and included in x

    elif included and excluded:
        return lambda x: isinstance(x, str) and included in x and all(aa not in x for aa in excluded)


    else:
        return lambda x: isinstance(x, str) and all(aa not in x for aa in excluded)


def identity_check(characters):
    """
    Returns a lambda function that checks if the sixth character of the identity value
    matches any of the given 'characters'.
    """
    if isinstance(characters, str):
        characters = set(characters)
    elif isinstance(characters, (list, tuple)):
        characters = set(characters)
    else:
        raise ValueError("characters must be a string, list, or tuple")

    return lambda x: isinstance(x, str) and len(x) > 6 and x[5] in characters

def nisv_check():
    """
    Returns a function that checks if the NISV_ERHALTEN_KONTR_VERLINKT value is NaN.
    """
    return pd.isnull

def kanton_check(required_kanton):
    """
    Returns a lambda function that checks if the 'KANTON' column matches the required value.
    """
    return lambda x: isinstance(x, str) and x == required_kanton




def doProcess(INPATH=".",OUTPATH="."):

    # CSVs
    # Define the input folder (relative to the parent directory of the script)
    #input_folder = os.path.join(os.path.dirname(os.getcwd()), "input")

    # Define file patterns for dynamic matching
    file_patterns = {
        "ASSET_PROPERTY_REPORT": r"ASSET_Ericsson_Report_PROPERTY_.*_Mo\.csv",
        "ASSET_2G_REPORT": r"Asset_2G Report\.csv",
        "ASSET_5G_REPORT": r"ASSET_Ericsson_Report_5G_.*_Mo\.csv",
        "ASSET_LTE_REPORT": r"ASSET_Ericsson_Report_LTE_.*_Mo\.csv",
        "ASSET_UMTS_REPORT": r"ASSET_Ericsson_Report_UMTS_.*_Mo\.csv",
        "ASSET_SUMMARY_REPORT": r"ASSET_Ericsson_Report_SUMMARY_.*_Mo\.csv",
        "ASSET_CARRIERS_REPORT": r"ASSET_Ericsson_Report_CARRIER_.*_Mo\.csv",
        "INDIGO_TABLE_DATA": r"indigo_.*\.csv",
    }

    # Find files dynamically in the input folder
    matched_files = {}
    for key, pattern in file_patterns.items():
        for file in os.listdir(INPATH):
            if re.fullmatch(pattern, file):
                matched_files[key] = os.path.join(INPATH, file)
                break
        else:
            matched_files[key] = None  # If no file matches, set None

    # Example: Access the paths dynamically
    ASSET_5G_FILE = matched_files.get("ASSET_5G_REPORT")
    ASSET_LTE_FILE = matched_files.get("ASSET_LTE_REPORT")
    ASSET_UMTS_FILE = matched_files.get("ASSET_UMTS_REPORT")
    ASSET_SUMMARY_FILE = matched_files.get("ASSET_SUMMARY_REPORT")
    ASSET_CARRIERS_FILE = matched_files.get("ASSET_CARRIERS_REPORT")
    INDIGO_TABLE_DATA_FILE = matched_files.get("INDIGO_TABLE_DATA")


    # Print results for verification
    for key, file_path in matched_files.items():
        print(f"{key}: {file_path if file_path else 'File not found!'}")

    # file with 4BCs in Wallis
    #WALLIS_4B = "wallis_4BCs.txt"
    WALLIS_4B_FILE = os.path.join(INPATH, "wallis_4BCs.txt")

    #WALLIS_4B_FILE = os.path.join(os.path.dirname(os.getcwd()), "input", WALLIS_4B)

    # # CSVs version
    asset_5g = pd.read_csv(ASSET_5G_FILE, header=0, low_memory=False, sep=';').dropna(subset=['Cell_Coverage'])
    asset_4g = pd.read_csv(ASSET_LTE_FILE, header=0, low_memory=False, sep=';').dropna(subset=['Cell_Coverage'])
    asset_3g = pd.read_csv(ASSET_UMTS_FILE, header=0, low_memory=False, sep=';').dropna(subset=['Cell_Coverage'])
    asset_summary = pd.read_csv(ASSET_SUMMARY_FILE, header=0, low_memory=False, sep=';').dropna(subset=['Cell_Coverage'])
    asset_carriers = pd.read_csv(ASSET_CARRIERS_FILE, header=0, low_memory=False, sep=';').groupby(NODE_IDENTITY)['Carrier ID'].apply(list).to_dict()
    indigo = pd.read_csv(INDIGO_TABLE_DATA_FILE, low_memory=False).dropna(subset=['ANTSUPPLYTYPE_CODE'])

    wallis_4bc = pd.read_csv(WALLIS_4B_FILE, header=None, low_memory=False, names=['4BC'])

     

    """### MAIN"""

    
    #------- 5G -----------------------------------------------------------------------------------------------------------------------------------------------------------
    corrections_5g = list()

    # splitting input asset file (important to keep same index)
    asset_out_5g = get_asset_outdoor_cells(asset_5g)
    asset_out_5g[CARRIER_ON_NODE_LEVEL] = 'Carrier Available'

    antenna_mask = asset_out_5g['Antenna'].apply(is_active)

    x_cell_mask = asset_out_5g['Identity'].str[5] == 'X'
    combined_mask = antenna_mask & x_cell_mask


    asset_5g_merged = merge_asset_5g_with_indigo(asset_out_5g, concat_cell_names_indigo(indigo))
    asset_out_planned = get_asset_planned_cells(asset_out_5g, indigo)

    # Noise Figure
    col_name = ColumnName('Noise Figure (dB)')

    corrections_5g += simple_correction_new(asset_out_5g[combined_mask], col_name, 6)
    corrections_5g += simple_correction_new(asset_out_5g[~combined_mask], col_name, 2)
    #corrections_5g += simple_correction(asset_out_5g[combined_mask], 'Noise Figure (dB)', 6)
    #corrections_5g += simple_correction(asset_out_5g[~combined_mask], 'Noise Figure (dB)', 2)

    # Power Offsets
    antenna_mask_planned = asset_out_planned['Antenna'].apply(is_active)

    x_cell_mask_planned = asset_out_5g['Identity'].str[5] == 'X'
    combined_mask_planned = antenna_mask_planned & x_cell_mask_planned

    planned_x_active = asset_out_planned[combined_mask_planned]
    planned_not_x_active = asset_out_planned[~combined_mask_planned]
    
    col_name = ColumnName('PSS EPRE Offset (dB)')
    corrections_5g += simple_correction_new(planned_x_active, col_name, 0)
    corrections_5g += simple_correction_new(planned_not_x_active, col_name, 0)
    
    col_name = ColumnName('PDCCH EPRE Offset (dB)')
    corrections_5g += simple_correction_new(planned_x_active, col_name, 0)
    corrections_5g += simple_correction_new(planned_not_x_active, col_name, 0)
    
    col_name = ColumnName('PDSCH EPRE Offset (dB)')
    corrections_5g += simple_correction_new(planned_x_active, col_name, -7)
    corrections_5g += simple_correction_new(planned_not_x_active,col_name, 0)

    col_name = ColumnName('CSI-RS EPRE Offset (dB)')
    corrections_5g += simple_correction_new(planned_x_active, col_name, 0)
    corrections_5g += simple_correction_new(planned_not_x_active, col_name, 0)

    # Modulation Types
    corrections_5g += simple_correction(asset_out_5g, 'DL Highest Supported Modulation', '256QAM')

    # PDSCH Overhead (%)
    corrections_5g += simple_correction(asset_out_5g, 'PDSCH Overhead (%)', 14)

    # PUSCH Overhead (%)
    corrections_5g += simple_correction(asset_out_5g, 'PUSCH Overhead (%)', 8)

    # Cell Range
    corrections_5g += simple_correction(asset_out_5g, 'Threshold Timing Advance Max Range (km)', 2.43, Cell_Network='5G3600')

    # SSB Periodicity
    corrections_5g += simple_correction(asset_out_5g, 'SSB Periodicity (ms)', 20)

    # SSB Numerology
    col_name = ColumnName('SSB Numerology')
    corrections_5g += simple_correction_new(asset_out_5g, col_name, 1, Cell_Network='5G3600')
    corrections_5g += simple_correction_new(asset_out_5g, col_name, 0, Cell_Network='5G700')
    corrections_5g += simple_correction_new(asset_out_5g, col_name, 0, Cell_Network='5G2100')
    corrections_5g += simple_correction_new(asset_out_5g, col_name, 0, Cell_Network='5G1800')

    # DL Numerology
    col_name = ColumnName('DL Numerology')
    corrections_5g += simple_correction_new(asset_out_5g, col_name, 1, Cell_Network='5G3600')
    corrections_5g += simple_correction_new(asset_out_5g, col_name, 0, Cell_Network='5G700')
    corrections_5g += simple_correction_new(asset_out_5g, col_name, 0, Cell_Network='5G2100')
    corrections_5g += simple_correction_new(asset_out_5g, col_name, 0, Cell_Network='5G1800')

    # UL Numerology
    col_name = ColumnName('DL Numerology')
    corrections_5g += simple_correction_new(asset_out_5g, col_name, 0, Cell_Network='5G700')
    corrections_5g += simple_correction_new(asset_out_5g, col_name, 0, Cell_Network='5G2100')
    corrections_5g += simple_correction_new(asset_out_5g, col_name, 0, Cell_Network='5G1800')

    # TDD Slot Parameters UL/DL
    corrections_5g += simple_correction(asset_out_5g, 'TDD Slot DL (%)', 74.00, Cell_Network='5G3600')
    corrections_5g += simple_correction(asset_out_5g, 'TDD Slot UL (%)', 20.00, Cell_Network='5G3600')

    # DL Cyclic prefix
    corrections_5g += simple_correction(asset_out_5g, 'DL Cyclic prefix', 'Normal')

    # Asset Design Activation
    #corrections_5g += simple_correction(asset_out_5g, 'ASSET Design Activation', 'Yes')

    # Beam set
    # corrections_5g += simple_correction(asset_out_5g[combined_mask], 'Beam Set ID', '3600_M_B+4_Tall', =)
    '''corrections_5g += simple_correction(asset_out_5g[combined_mask], 'Beam Set ID', '3600_M_B+4_Tall', **{'Height-Inv (m)': ('lt', 20.00)})
    corrections_5g += simple_correction(asset_out_5g[combined_mask], 'Beam Set ID', '3600_M_B+5_Tall', **{'Height-Inv (m)': ('range', (20.00, 30.00))})
    corrections_5g += simple_correction(asset_out_5g[combined_mask], 'Beam Set ID', '3600_M_B+6_Tall', **{'Height-Inv (m)': ('gt', 30.00)})'''

    # Assigned Carrier
    col_name = ColumnName('Assigned Carrier')
    corrections_5g += simple_correction_new(asset_out_5g, col_name, 'N700_15M_156100', Identity=lambda x: len(x) > 5 and x[5] == 'Q')
    corrections_5g += simple_correction_new(asset_out_5g, col_name, 'N2100_20M_426060', Identity=lambda x: len(x) > 5 and x[5] == 'U')


    wallis_4bc_set = set(wallis_4bc['4BC'].tolist())  # Convert to a set for fast lookup

    # Assigned Carrier for X cells NOT in wallis_4bc
    corrections_5g += simple_correction_new(
        asset_out_5g,
        col_name,
        'N3600_100M_643332',
        Identity=lambda x: len(x) > 5 and x[5] == 'X' and x[:4] not in wallis_4bc_set
    )

    # Assigned Carrier for X cells IN wallis_4bc
    corrections_5g += simple_correction_new(
        asset_out_5g,
        col_name,
        'N3600_40M_641332',
        Identity=lambda x: len(x) > 5 and x[5] == 'X' and x[:4] in wallis_4bc_set
    )


    '''corrections_5g += simple_correction(
       asset_5g_merged, 'Max TX Power (dBm)', 46.0,
       Identity=identity_check('X'), Antenna=antenna_check('AIR3268'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 46.0,
        Identity=identity_check('X'), Antenna=antenna_check('AIR3278B78'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 46.0,
        Identity=identity_check('X'), Antenna=antenna_check('Hybrid'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 46.0,
        Identity=identity_check('X'), Antenna=antenna_check('AIR3239B78'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 46.0,
        Identity=identity_check('X'), Antenna=antenna_check('AIR6488B43'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 38.5,
        Identity=identity_check('X'), Antenna=antenna_check('InterAIR'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 38.5,
        Identity=identity_check('X'), Antenna=antenna_check('AIR3218'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 44.5,
        Identity=identity_check('X'), Antenna=antenna_check('AOC'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 44.0,
        Identity=identity_check('X'), Antenna=antenna_check('A11'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 45.5,
        Identity=identity_check('X'), Antenna=antenna_check('6313'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 44,
        Identity=identity_check('X'), Antenna=antenna_check(excluded=["AIR3268", "Hybrid", "InterAIR", "AIR3218","AOC","A11","6313","AIR"]), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    #Q-CELL

    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 44.0,
        Identity=identity_check('Q'), Antenna=antenna_check('A11'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 44.0,
        Identity=identity_check('Q'), Antenna=antenna_check('AIR3278B78'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 44.0,
        Identity=identity_check('Q'), Antenna=antenna_check('AIR3268'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 44.0,
        Identity=identity_check('Q'), Antenna=antenna_check('Hybrid'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 44.0,
        Identity=identity_check('Q'), Antenna=antenna_check('AIR3239B78'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 44.0,
        Identity=identity_check('Q'), Antenna=antenna_check('AIR6488B43'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 44.23,
        Identity=identity_check('Q'), Antenna=antenna_check('InterAIR'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 43.0,
        Identity=identity_check('Q'), Antenna=antenna_check('AOC'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 44.23,
        Identity=identity_check('Q'), Antenna=antenna_check('AIR3218'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 43,
        Identity=identity_check('Q'), Antenna=antenna_check('AHP4518R4'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 44.5,
        Identity=identity_check('Q'), Antenna=antenna_check('AHP4518R3'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 44.5,
        Identity=identity_check('Q'), Antenna=antenna_check('6313-P'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 43.0,
        Identity=identity_check('Q'), Antenna=antenna_check('80011878'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 43.0,
        Identity=identity_check('Q'), Antenna=antenna_check(excluded=["AIR3268", "Hybrid", "InterAIR", "AIR3218","AOC","A11","6313-P","AHP4518R4","AHP4518R3","80011878","AIR"]), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    #U-CELL(2100)


    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 47.0,
        Identity=identity_check('U'), Antenna=antenna_check('A11'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 46.0,
        Identity=identity_check('U'), Antenna=antenna_check('AIR3278B78'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 46.0,
        Identity=identity_check('U'), Antenna=antenna_check('AIR3239B78'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 46.0,
        Identity=identity_check('U'), Antenna=antenna_check('AIR3268'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 46.0,
        Identity=identity_check('U'), Antenna=antenna_check('Hybrid'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 46.0,
        Identity=identity_check('U'), Antenna=antenna_check('AIR6488B43'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 43.9,
        Identity=identity_check('U'), Antenna=antenna_check('InterAIR'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 47.5,
        Identity=identity_check('U'), Antenna=antenna_check('AOC'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 43.9,
        Identity=identity_check('U'), Antenna=antenna_check('AIR3218'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 47.5,
        Identity=identity_check('U'), Antenna=antenna_check('AHP4518R4'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 47.5,
        Identity=identity_check('U'), Antenna=antenna_check('AHP4518R3'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 47.0,
        Identity=identity_check('U'), Antenna=antenna_check('6313-P'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 47.0,
        Identity=identity_check('U'), Antenna=antenna_check('80011878'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_5g += simple_correction(
        asset_5g_merged, 'Max TX Power (dBm)', 45,
        Identity=identity_check('U'), Antenna=antenna_check(excluded=["AIR3268", "Hybrid", "InterAIR", "AIR3218","AOC","A11","6313-P","AHP4518R4","AHP4518R3","80011878","AIR"]), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )'''

    # Save corrections
    output_5g = get_corrections(asset_5g, corrections_5g)
    output_5g = output_5g[output_5g['Corrections'].apply(lambda x: x != [])]

    # Additional logic for Assigned Carrier on Node level
    output_5g = add_carrier_on_node_level(output_5g, asset_carriers)

        

    #------- 4G -----------------------------------------------------------------------------------------------------------------------------------------------------------

    
    corrections_4g = list()
    asset_4g_out = get_asset_outdoor_cells(asset_4g)
    # asset_4g_out['Carrier on node level'] = 'Carrier Available'

    asset_4g_merged = merge_asset_4g_with_indigo(asset_4g_out, concat_cell_names_indigo(indigo))

    asset_4g_out_planned = get_asset_planned_cells(asset_4g_merged, indigo)
    asset_4g_merged.to_csv("asset_test_4g.csv")
    # Noise Figure
    corrections_4g += simple_correction(asset_4g_merged, 'Noise Figure (dB)', 3, PREAMPLIFIER_CODE='kein TMA/ASC')

    # Power Offsets
    for net_id in ['Z', 'P', 'C']:
        corrections_4g += simple_correction(asset_4g_out_planned, 'Traffic Offset (dB)', 0.01, NET_IDS=net_id)
        corrections_4g += simple_correction(asset_4g_out_planned, 'Control Offset (dB)', 0.01, NET_IDS=net_id)
        corrections_4g += simple_correction(asset_4g_out_planned, 'Synchronisation Offset (dB)', 0.01, NET_IDS=net_id)
        corrections_4g += simple_correction(asset_4g_out_planned, 'Broadcast Offset (dB)', 0.01, NET_IDS=net_id)
    for net_id in ['K', 'O', 'R', 'M', 'N', 'V', 'W']:
        corrections_4g += simple_correction(asset_4g_out_planned, 'Traffic Offset (dB)', 3.01, NET_IDS=net_id)
        corrections_4g += simple_correction(asset_4g_out_planned, 'Control Offset (dB)', 3.01, NET_IDS=net_id)
        corrections_4g += simple_correction(asset_4g_out_planned, 'Synchronisation Offset (dB)', 3.01, NET_IDS=net_id)
        corrections_4g += simple_correction(asset_4g_out_planned, 'Broadcast Offset (dB)', 3.01, NET_IDS=net_id)

    # Modulation Type
    corrections_4g += simple_correction(asset_4g_merged, 'DL Highest Supported Modulation', '256QAM')

    # Asset Design Activation
    corrections_4g += simple_correction(asset_4g_merged, 'ASSET Design Activation', 'Yes')


    # Assigned Carrier
    col_name = ColumnName('Assigned Carrier')
    corrections_4g += simple_correction_new(asset_4g_merged, col_name, 'L700_15M_9435', Identity=lambda x: len(x) > 5 and x[5] == 'Z')
    corrections_4g += simple_correction_new(asset_4g_merged, col_name, 'L800_10M_6300', Identity=lambda x: len(x) > 5 and x[5] == 'P')
    # corrections_4g += simple_correction(asset_4g_merged, 'Assigned Carrier', 'L800_15M_CERN_Carrier1_6225', Identity=lambda x: len(x) > 5 and x[5] == 'P')
    # corrections_4g += simple_correction(asset_4g_merged, 'Assigned Carrier', 'L800_15M_CERN_Carrier2_6225', Identity=lambda x: len(x) > 5 and x[5] == 'P')
    corrections_4g += simple_correction_new(asset_4g_merged, col_name, 'L900_10M_3750', Identity=lambda x: len(x) > 5 and x[5] == 'C')
    corrections_4g += simple_correction_new(asset_4g_merged, col_name, 'L1400_20M_10170', Identity=lambda x: len(x) > 5 and x[5] == 'K')
    corrections_4g += simple_correction_new(asset_4g_merged, col_name, 'L1800_20M_1301', Identity=lambda x: len(x) > 5 and x[5] == 'O')
    corrections_4g += simple_correction_new(asset_4g_merged, col_name, 'L1800_10M_1445', Identity=lambda x: len(x) > 5 and x[5] == 'R')


    # Boarder Coordination
    # corrections_4g += simple_correction(asset_4g_merged, 'Assigned Carrier', 'L2100_10M_203', **{'Boarder_Coordination': ('neq', 'CH-FL')}, Identity=lambda x: len(x) > 5 and x[5] == 'M')
    # corrections_4g += simple_correction(asset_4g_merged, 'Assigned Carrier', 'L2100_10M_253', KANTON='Liechtenstein', Identity=lambda x: len(x) > 5 and x[5] == 'M')


    # Define identity function using the dictionary
    corrections_4g += simple_correction_new(
        asset_4g_merged, col_name, 'L2100_10M_253',
        Identity=lambda x: len(x) > 5 and x[5] == 'M',
        KANTON=kanton_check('Liechtenstein')  # Ensures only 'Lichtenstein' is used
    )

    # Apply correction for non-Lichtenstein cases
    corrections_4g += simple_correction_new(
        asset_4g_merged, col_name, 'L2100_20M_203',
        Identity=lambda x: len(x) > 5 and x[5] == 'M',
        KANTON=lambda x: isinstance(x, str) and x != 'Liechtenstein'  # Ensures any other kanton is used
    )

    corrections_4g += simple_correction_new(asset_4g_merged, col_name, 'L2600_20M_3100', Identity=lambda x: len(x) > 5 and x[5] == 'N')
    corrections_4g += simple_correction_new(asset_4g_merged, col_name, 'L2600TDD_20M_37900', Identity=lambda x: len(x) > 5 and x[5] == 'V')
    corrections_4g += simple_correction_new(asset_4g_merged, col_name, 'L2600TDD_20M_38098', Identity=lambda x: len(x) > 5 and x[5] == 'W')


    # # Max Tx Power (dBm)


    '''corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 44.0,
        Identity=identity_check('Z'), Antenna=antenna_check('A11'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 44.0,
        Identity=identity_check('Z'), Antenna=antenna_check('AIR3278B78'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 44.0,
        Identity=identity_check('Z'), Antenna=antenna_check('AIR3268'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 44.0,
        Identity=identity_check('Z'), Antenna=antenna_check('Hybrid'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 44.0,
        Identity=identity_check('Z'), Antenna=antenna_check('AIR3239B78'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 44.0,
        Identity=identity_check('Z'), Antenna=antenna_check('AIR6488B43'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 44.23,
        Identity=identity_check('Z'), Antenna=antenna_check('InterAIR'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 43.0,
        Identity=identity_check('Z'), Antenna=antenna_check('AOC'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 44.23,
        Identity=identity_check('Z'), Antenna=antenna_check('AIR3218'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 43,
        Identity=identity_check('Z'), Antenna=antenna_check('AHP4518R4'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 44.5,
        Identity=identity_check('Z'), Antenna=antenna_check('AHP4518R3'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 44.5,
        Identity=identity_check('Z'), Antenna=antenna_check('6313-P'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 43.0,
        Identity=identity_check('Z'), Antenna=antenna_check('80011878'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 43.0,
        Identity=identity_check('Z'), Antenna=antenna_check(excluded=["AIR3268", "Hybrid", "InterAIR", "AIR3218","AOC","A11","6313-P","AHP4518R4","AHP4518R3","80011878","AIR"]), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )


    # 800 BAND
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 43.5,
        Identity=identity_check('P'), Antenna=antenna_check('A11'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 42.0,
        Identity=identity_check('P'), Antenna=antenna_check('AIR3278B78'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 42.0,
        Identity=identity_check('P'), Antenna=antenna_check('AIR3239B78'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 42.0,
        Identity=identity_check('P'), Antenna=antenna_check('AIR3268'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 42.0,
        Identity=identity_check('P'), Antenna=antenna_check('Hybrid'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 42.0,
        Identity=identity_check('P'), Antenna=antenna_check('AIR6488B43'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 42.42,
        Identity=identity_check('P'), Antenna=antenna_check('InterAIR'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 42.5,
        Identity=identity_check('P'), Antenna=antenna_check('AOC'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 42.42,
        Identity=identity_check('P'), Antenna=antenna_check('AIR3218'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 42,
        Identity=identity_check('P'), Antenna=antenna_check('AHP4518R4'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 46.5,
        Identity=identity_check('P'), Antenna=antenna_check('AHP4518R3'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 43,
        Identity=identity_check('P'), Antenna=antenna_check('6313-P'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 42,
        Identity=identity_check('P'), Antenna=antenna_check('80011878'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 43,
        Identity=identity_check('P'), Antenna=antenna_check(excluded=["AIR3268", "Hybrid", "InterAIR", "AIR3218","AOC","A11","6313-P","AHP4518R4","AHP4518R3","80011878","AIR"]), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    # 1400 BAND
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 41.0,
        Identity=identity_check('K'), Antenna=antenna_check('A11'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 41.0,
        Identity=identity_check('K'), Antenna=antenna_check('AIR3278B78'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 41.0,
        Identity=identity_check('K'), Antenna=antenna_check('AIR3239B78'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 41.0,
        Identity=identity_check('K'), Antenna=antenna_check('AIR3268'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 41.0,
        Identity=identity_check('K'), Antenna=antenna_check('Hybrid'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 41.0,
        Identity=identity_check('P'), Antenna=antenna_check('AIR6488B43'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )


    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', pd.NA,
        Identity=identity_check('K'), Antenna=antenna_check('AIR3218'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', pd.NA,
        Identity=identity_check('K'), Antenna=antenna_check('InterAIR'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 40.5,
        Identity=identity_check('K'), Antenna=antenna_check('AOC'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )


    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 40.5,
        Identity=identity_check('K'), Antenna=antenna_check('AHP4518R4'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 40.5,
        Identity=identity_check('K'), Antenna=antenna_check('AHP4518R3'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )


    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 40.5,
        Identity=identity_check('K'), Antenna=antenna_check('80011878'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 41,
        Identity=identity_check('K'), Antenna=antenna_check(excluded=["AIR3268", "Hybrid", "InterAIR", "AIR3218","AOC","A11","AHP4518R4","AHP4518R3","80011878","AIR"]), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )


    # 1800 BAND
    #O & R- CELL
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 46.5,
        Identity=identity_check(['O','R']), Antenna=antenna_check('A11'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 46.0,
        Identity=identity_check(['O','R']), Antenna=antenna_check('AIR3278B78'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 46.0,
        Identity=identity_check(['O','R']), Antenna=antenna_check('AIR3239B78'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 46.0,
        Identity=identity_check(['O','R']), Antenna=antenna_check('AIR3268'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 46.0,
        Identity=identity_check(['O','R']), Antenna=antenna_check('Hybrid'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 46.0,
        Identity=identity_check(['O','R']), Antenna=antenna_check('AIR6488B43'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 44.28,
        Identity=identity_check(['O','R']), Antenna=antenna_check('InterAIR'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 47,
        Identity=identity_check(['O','R']), Antenna=antenna_check('AOC'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 44.28,
        Identity=identity_check(['O','R']), Antenna=antenna_check('AIR3218'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 47,
        Identity=identity_check(['O','R']), Antenna=antenna_check('AHP4518R4'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 47,
        Identity=identity_check(['O','R']), Antenna=antenna_check('AHP4518R3'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 46.5,
        Identity=identity_check(['O','R']), Antenna=antenna_check('6313-P'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 46.5,
        Identity=identity_check(['O','R']), Antenna=antenna_check('80011878'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 44,
        Identity=identity_check(['O','R']), Antenna=antenna_check(excluded=["AIR3268", "Hybrid", "InterAIR", "AIR3218","AOC","A11","6313-P","AHP4518R4","AHP4518R3","80011878","AIR"]), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    # 2100 BAND
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 47.0,
        Identity=identity_check('M'), Antenna=antenna_check('A11'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 46.0,
        Identity=identity_check('M'), Antenna=antenna_check('AIR3278B78'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 46.0,
        Identity=identity_check('M'), Antenna=antenna_check('AIR3239B78'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 46.0,
        Identity=identity_check('M'), Antenna=antenna_check('AIR3268'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 46.0,
        Identity=identity_check('M'), Antenna=antenna_check('Hybrid'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 46.0,
        Identity=identity_check('M'), Antenna=antenna_check('AIR6488B43'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 43.9,
        Identity=identity_check('M'), Antenna=antenna_check('InterAIR'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 47.5,
        Identity=identity_check('M'), Antenna=antenna_check('AOC'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 43.9,
        Identity=identity_check('M'), Antenna=antenna_check('AIR3218'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 47.5,
        Identity=identity_check('M'), Antenna=antenna_check('AHP4518R4'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 47.5,
        Identity=identity_check('M'), Antenna=antenna_check('AHP4518R3'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 47.0,
        Identity=identity_check('M'), Antenna=antenna_check('6313-P'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 47.0,
        Identity=identity_check('M'), Antenna=antenna_check('80011878'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 45,
        Identity=identity_check('M'), Antenna=antenna_check(excluded=["AIR3268", "Hybrid", "InterAIR", "AIR3218","AOC","A11","6313-P","AHP4518R4","AHP4518R3","80011878","AIR"]), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )


    # 2600 BAND
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 43.0,
        Identity=identity_check(['N','V','W']), Antenna=antenna_check('A11'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 42.0,
        Identity=identity_check(['N','V','W']), Antenna=antenna_check('AIR3278B78'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 42.0,
        Identity=identity_check(['N','V','W']), Antenna=antenna_check('AIR3239B78'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 42.0,
        Identity=identity_check(['N','V','W']), Antenna=antenna_check('AIR3268'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 42.0,
        Identity=identity_check(['N','V','W']), Antenna=antenna_check('Hybrid'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 42.0,
        Identity=identity_check(['N','V','W']), Antenna=antenna_check('AIR6488B43'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 39.12,
        Identity=identity_check(['N','V','W']), Antenna=antenna_check('InterAIR'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 43.0,
        Identity=identity_check(['N','V','W']), Antenna=antenna_check('AOC'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 39.12,
        Identity=identity_check(['N','V','W']), Antenna=antenna_check('AIR3218'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 43.0,
        Identity=identity_check(['N','V','W']), Antenna=antenna_check('AHP4518R4'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 43.0,
        Identity=identity_check(['N','V','W']), Antenna=antenna_check('AHP4518R3'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 43.5,
        Identity=identity_check(['N','V','W']), Antenna=antenna_check('6313-P'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 43.0,
        Identity=identity_check(['N','V','W']), Antenna=antenna_check('80011878'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 43.0,
        Identity=identity_check(['N','V','W']), Antenna=antenna_check(excluded=["AIR3268", "Hybrid", "InterAIR", "AIR3218","AOC","A11","6313-P","AHP4518R4","AHP4518R3","80011878","AIR"]), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    #L900-C cell
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 38.5,
        Identity=identity_check('C'), Antenna=antenna_check('A11'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 37.5,
        Identity=identity_check('C'), Antenna=antenna_check('AIR3278B78'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 37.5,
        Identity=identity_check('C'), Antenna=antenna_check('AIR3239B78'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 37.5,
        Identity=identity_check('C'), Antenna=antenna_check('AIR3268'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 37.5,
        Identity=identity_check('C'), Antenna=antenna_check('Hybrid'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 37.5,
        Identity=identity_check('C'), Antenna=antenna_check('AIR6488B43'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 37.45,
        Identity=identity_check('C'), Antenna=antenna_check('InterAIR'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 37.0,
        Identity=identity_check('C'), Antenna=antenna_check('AOC'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 37.45,
        Identity=identity_check('C'), Antenna=antenna_check('AIR3218'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 36.0,
        Identity=identity_check('C'), Antenna=antenna_check('AHP4518R4'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )
    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 37.5,
        Identity=identity_check('C'), Antenna=antenna_check('AHP4518R3'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 38.0,
        Identity=identity_check('C'), Antenna=antenna_check('6313-P'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 36.5,
        Identity=identity_check('C'), Antenna=antenna_check('80011878'), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )

    corrections_4g += simple_correction(
        asset_4g_merged, 'Maximum TX Power (dBm)', 38.0,
        Identity=identity_check('C'), Antenna=antenna_check(excluded=["AIR3268", "Hybrid", "InterAIR", "AIR3218","AOC","A11","6313-P","AHP4518R4","AHP4518R3","80011878","AIR"]), NISV_ERHALTEN_KONTR_VERLINKT=nisv_check()
    )'''

    
    # Save corrections
    output_4g = get_corrections(asset_4g, corrections_4g)
    output_4g = output_4g[output_4g['Corrections'].apply(lambda x: x != [])]


    # Additional logic for Assigned Carrier on Node level
    output_4g = add_carrier_on_node_level(output_4g, asset_carriers)

    output_4g
    #------- 3G -----------------------------------------------------------------------------------------------------------------------------------------------------------

    
    corrections_3g = list()
    asset_3g_out = get_asset_outdoor_cells(asset_3g)
    asset_3g_merged = merge_asset_3g_with_indigo(asset_3g_out, concat_cell_names_indigo(indigo))

    # Noise Figure
    corrections_3g += simple_correction(asset_3g_out, 'Noise Figure (dB)', 1.8)

    # Modulation Type
    #corrections_3g += simple_correction(asset_3g_out, 'HSDPA Max Supported Modulations', '64QAM')

    # Tx Power Limits
    #corrections_3g += simple_correction(asset_3g_out, 'Min DL Power per Connection (dBm)', -20)
    #corrections_3g += simple_correction(asset_3g_out, 'Max DL Power per Connection (dBm)', value=lambda row: row['Maximum TX Power (dBm)']-1)

    

    # Save corrections
    output_3g = get_corrections(asset_3g, corrections_3g)
    output_3g = output_3g[output_3g['Corrections'].apply(lambda x: x != [])]


    # Additional logic for Assigned Carrier on Node level
    output_3g = add_carrier_on_node_level(output_3g, asset_carriers)
        

    #------- SUMMARY -----------------------------------------------------------------------------------------------------------------------------------------------------------
    

    output_summary = asset_summary.copy()

    # Active Technology
    col_name = ColumnName('Active Technology')
    corrections_summary = simple_correction_new(output_summary,
                                            col_name,
                                            'UMTS',
                                            Identity=lambda x: len(x) > 5 and x[5] in ['A', 'G']
                                           )

    corrections_summary = simple_correction_new(output_summary,
                                            col_name,
                                            'LTE',
                                            Identity=lambda x: len(x) > 5 and x[5] in ['Z', 'P', 'C', 'K', 'O', 'R', 'M', 'N']
                                           )

    corrections_summary = simple_correction_new(output_summary,
                                            col_name,
                                            '5G',
                                            Identity=lambda x: len(x) > 5 and x[5] in ['Q', 'U', 'X']
                                           )

    # Cell Network
    col_name = ColumnName('Cell_Network')
    corrections_summary += simple_correction_new(output_summary, col_name, 'UMTS900', Identity=lambda x: len(x) > 5 and x[5] == 'A')
    corrections_summary += simple_correction_new(output_summary, col_name, 'UMTS2100', Identity=lambda x: len(x) > 5 and x[5] == 'G')

    corrections_summary += simple_correction_new(output_summary, col_name, 'LTE700', Identity=lambda x: len(x) > 5 and x[5] == 'Z')
    corrections_summary += simple_correction_new(output_summary, col_name, 'LTE800', Identity=lambda x: len(x) > 5 and x[5] == 'P')
    corrections_summary += simple_correction_new(output_summary, col_name, 'LTE900', Identity=lambda x: len(x) > 5 and x[5] == 'C')
    corrections_summary += simple_correction_new(output_summary, col_name, 'LTE1400', Identity=lambda x: len(x) > 5 and x[5] == 'K')
    corrections_summary += simple_correction_new(output_summary, col_name, 'LTE1800', Identity=lambda x: len(x) > 5 and x[5] == 'O')
    corrections_summary += simple_correction_new(output_summary, col_name, 'LTE2100', Identity=lambda x: len(x) > 5 and x[5] == 'M')
    corrections_summary += simple_correction_new(output_summary, col_name, 'LTE2600', Identity=lambda x: len(x) > 5 and x[5] == 'N')

    corrections_summary += simple_correction_new(output_summary, col_name, '5G700', Identity=lambda x: len(x) > 5 and x[5] == 'Q' )
    corrections_summary += simple_correction_new(output_summary, col_name, '5G2100', Identity=lambda x: len(x) > 5 and x[5] == 'U')
    corrections_summary += simple_correction_new(output_summary, col_name, '5G3600', Identity=lambda x: len(x) > 5 and x[5] == 'X', **{'Cell_Network': ('neq', '5G1800')}) # to be checked NOT SWISSCOM

    # Save corrections
    output_summary = get_corrections(output_summary, corrections_summary)
    output_summary = output_summary[output_summary['Corrections'].apply(lambda x: x != [])]

        




    # OUTPUT COLUMNS
    OUTPUT_5G_COLUMNS = ['Identity', 'Node IDENTITY', 'Parent ID', 'Property ID', 'Antenna', 'Designer',
                      'Noise Figure (dB)', 'PSS EPRE Offset (dB)', 'PDCCH EPRE Offset (dB)', 'PDSCH EPRE Offset (dB)', 'CSI-RS EPRE Offset (dB)',
                      'DL Highest Supported Modulation', 'SSB Numerology', 'DL Numerology', 'UL Numerology', 'TDD Slot DL (%)', 'TDD Slot UL (%)',
                      'Threshold Timing Advance Max Range (km)', 'ASSET Design Activation', 'Beam Set ID', 'Max TX Power (dBm)',
                         'PDSCH Overhead (%)', 'PUSCH Overhead (%)', 'Assigned Carrier', 'Carrier on node level',
                     ]

    OUTPUT_4G_COLUMNS = ['Identity','Node IDENTITY',  'Parent ID', 'Property ID', 'Antenna', 'Designer',
                         'Noise Figure (dB)', 'Traffic Offset (dB)', 'Control Offset (dB)', 'Synchronisation Offset (dB)', 'Broadcast Offset (dB)', 'Maximum TX Power (dBm)',
                      'DL Highest Supported Modulation', 'Assigned Carrier', 'Carrier on node level',
                     ]

    OUTPUT_3G_COLUMNS = ['Identity', 'Node IDENTITY', 'Parent ID', 'Property ID', 'Antenna', 'Designer',
                         'Noise Figure (dB)', 'HSDPA Max Supported Modulations', 'Maximum TX Power (dBm)',
                         'Min DL Power per Connection (dBm)', 'Max DL Power per Connection (dBm)',
                     ]

    OUTPUT_SUMMARY_COLUMNS = ['Identity', 'Node IDENTITY', 'Parent ID', 'Property ID',
                              'Active Technology', 'Cell_Network',
                              'Primary Prediction Radius (km)', 'Secondary Prediction Radius (km)', 'Designer',
                             ]

    # Saving outputs
    current_date = datetime.now().date()

    # output_dir = os.path.join(os.path.dirname(os.getcwd()), "output", "health_check")
    # os.makedirs(output_dir, exist_ok=True)


    # with pd.ExcelWriter(
        # os.path.join(output_dir, f"health_check_{current_date}.xlsx"),
        # engine="openpyxl"
    # ) as writer:
        # output_5g.to_excel(writer, sheet_name='5G', index=False, columns=OUTPUT_5G_COLUMNS + ['Corrections'])
        # output_4g.to_excel(writer, sheet_name='LTE', index=False, columns=OUTPUT_4G_COLUMNS + ['Corrections'])
        # output_3g.to_excel(writer, sheet_name='UMTS', index=False, columns=OUTPUT_3G_COLUMNS + ['Corrections'])
        # output_summary.to_excel(writer, sheet_name='SUMMARY', index=False, columns=OUTPUT_SUMMARY_COLUMNS + ['Corrections'])
        
    output_file = os.path.join(OUTPATH, f"health_check_{current_date}.xlsx")

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        output_5g.to_excel(writer, sheet_name='5G', index=False, columns=OUTPUT_5G_COLUMNS + ['Corrections'])
        output_4g.to_excel(writer, sheet_name='LTE', index=False, columns=OUTPUT_4G_COLUMNS + ['Corrections'])
        output_3g.to_excel(writer, sheet_name='UMTS', index=False, columns=OUTPUT_3G_COLUMNS + ['Corrections'])
        output_summary.to_excel(writer, sheet_name='SUMMARY', index=False, columns=OUTPUT_SUMMARY_COLUMNS + ['Corrections'])


working_dir = os.path.dirname(os.path.abspath(__file__)) 
INPATH= working_dir
OUTPATH= working_dir   
doProcess(INPATH=INPATH,OUTPATH=OUTPATH)

