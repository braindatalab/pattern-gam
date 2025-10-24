import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression 
from scipy.interpolate import interp1d
from datetime import timedelta
import warnings
import pickle 
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches # Added for legend
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.impute import SimpleImputer
import os 
import seaborn as sns # For KDE fallback

# Suppress specific PerformanceWarnings from pandas about fragmentation
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# --- Configuration ---
data_path = './data/mimiciv/2.0' 
hosp_path = os.path.join(data_path, 'hosp')
icu_path = os.path.join(data_path, 'icu') 
output_path = './data/mimiciv' 

os.makedirs(output_path, exist_ok=True)
os.makedirs('./figures', exist_ok=True)

# --- File Paths ---
admissions_file = os.path.join(hosp_path, 'admissions.csv.gz')
patients_file = os.path.join(hosp_path, 'patients.csv.gz')
diagnoses_file = os.path.join(hosp_path, 'diagnoses_icd.csv.gz')
prescriptions_file = os.path.join(hosp_path, 'prescriptions.csv.gz')
labevents_file = os.path.join(hosp_path, 'labevents.csv.gz')
chartevents_file = os.path.join(icu_path, 'chartevents.csv.gz') 
icu_stays_file = os.path.join(icu_path, 'icustays.csv.gz') # Needed for chartevents mapping

# --- ItemID Definitions ---
vital_itemids = { 
    'GCS_Eye': 220739,'GCS_Verbal': 223900,'GCS_Motor': 223901,'Heart_Rate': 220045,
    'Sys_BP': 220179,'Dias_BP': 220180,'MAP': 220181,'Resp_Rate': 220210,
    'Temp_C': 223761,'SpO2': 220277,}
lab_itemids = { 
    'Sodium': 50983, 'Potassium': 50971, 'Chloride': 50902, 'Bicarbonate': 50882,
    'BUN': 51006, 'Creatinine': 50912, 'Glucose': 50931, 'WBC': 51301,
    'Hemoglobin': 51222, 'Hematocrit': 51221, 'Platelets': 51265,
    'Bilirubin_Total': 50885, 'ALT': 50861, 'AST': 50878, 'Lactate': 50813,
    'pH': 50820, 'Anion_Gap': 50868,}


# --- Helper Function for Value Counts ---
def print_value_counts(df, feature_list, stage_name, top_n=20):
    """Prints value counts or descriptions for features in a DataFrame."""
    print(f"\n--- Value Counts / Distribution ({stage_name}) ---")
    if df is None or df.empty:
        print("DataFrame is empty or None.")
        return

    for feature in feature_list:
        if feature in df.columns:
            col_data = df[feature]
            print(f"\nFeature: '{feature}'")
            if pd.api.types.is_numeric_dtype(col_data):
                # Check if effectively categorical based on unique values
                unique_count = col_data.nunique()
                if unique_count <= max(top_n, 30): # Treat as categorical if few unique values
                     print(f"  (Numeric treated as Categorical - {unique_count} unique values):")
                     counts = col_data.value_counts(dropna=False).sort_index()
                     print(f"  Counts (Top {top_n}):\n{counts.head(top_n)}")
                     if len(counts) > top_n: print("    ...")
                else:
                    # Continuous: Print describe() and maybe rounded counts
                    print(f"  (Continuous - {unique_count} unique values):")
                    print(f"  Description:\n{col_data.describe().to_string()}")
                    try:
                        # Rounding depends on expected scale
                        if col_data.abs().max() > 100 or col_data.abs().min() < 0.1 and col_data.abs().min() != 0:
                             rounded_data = col_data.round(1) # Round to 1dp for typical labs/vitals
                        else:
                             rounded_data = col_data.round(0) # Round to integer for age, counts etc.
                        counts = rounded_data.value_counts(dropna=False).sort_index()
                        print(f"  Approx Counts (Rounded, Top {top_n}):\n{counts.head(top_n)}")
                        if len(counts) > top_n: print("    ...")
                    except Exception as e:
                        print(f"    Could not generate rounded counts: {e}")

            elif pd.api.types.is_categorical_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
                print("  (Categorical/Object):")
                counts = col_data.value_counts(dropna=False).sort_index()
                print(f"  Counts (Top {top_n}):\n{counts.head(top_n)}")
                if len(counts) > top_n: print("    ...")
            else:
                print(f"  (Unknown type: {col_data.dtype})")
                print(f"  Description:\n{col_data.describe().to_string()}") # Try describe anyway
        else:
            print(f"\nFeature: '{feature}' - Not found in DataFrame at this stage.")
    print("--- End Value Counts ---")

# --- Helper Function for Aggregation (Unchanged from previous working version) ---
def load_and_aggregate_mean(filepath, relevant_itemids, admissions_df, time_col='charttime', value_col='valuenum', id_col='hadm_id', window_hours=24):
    # ... (Keep the function from the 2024-05-01T10:35:34.807Z response - calculating mean only) ...
    if not os.path.exists(filepath): return None
    print(f"Loading {os.path.basename(filepath)}...")
    try: # Simplified loading part for brevity, assume it worked before
        cols_to_load = ['itemid', time_col, value_col]
        if id_col == 'stay_id': cols_to_load.append('stay_id')
        else: cols_to_load.append('hadm_id')
        df_events = pd.read_csv(filepath, compression='gzip',usecols=cols_to_load,parse_dates=[time_col])
        df_events[value_col] = pd.to_numeric(df_events[value_col], errors='coerce')
        df_events.dropna(subset=[value_col, time_col], inplace=True)
    except Exception as e: print(f"Error loading {filepath}: {e}"); return None
    
    df_events = df_events[df_events['itemid'].isin(relevant_itemids.values())]
    if df_events.empty: return None
    itemid_to_name = {v: k for k, v in relevant_itemids.items()}
    df_events['feature_name'] = df_events['itemid'].map(itemid_to_name)
    id_col_final = id_col
    if id_col == 'stay_id':
        if not os.path.exists(icu_stays_file): print("Error: icustays.csv.gz needed."); return None
        df_icustays = pd.read_csv(icu_stays_file, usecols=['hadm_id', 'stay_id'])
        df_events = pd.merge(df_events, df_icustays, on='stay_id', how='inner')
        id_col_final = 'hadm_id'
        if df_events.empty: print("Warning: No matching stay_ids."); return None
    
    df_events = pd.merge(df_events, admissions_df[[id_col_final, 'admittime']], on=id_col_final, how='left')
    df_events.dropna(subset=['admittime'], inplace=True)
    df_events['time_delta'] = df_events[time_col] - df_events['admittime']
    time_window = timedelta(hours=window_hours)
    df_filtered = df_events[(df_events['time_delta'] >= timedelta(0)) & (df_events['time_delta'] <= time_window)]
    if df_filtered.empty: return None
    
    aggregations = {value_col: ['mean']}
    df_agg = df_filtered.groupby([id_col_final, 'feature_name']).agg(aggregations)
    df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
    df_agg = df_agg.reset_index()
    value_col_mean = f"{value_col}_mean"
    df_pivot = df_agg.pivot(index=id_col_final, columns='feature_name', values=value_col_mean)
    df_pivot.columns = [f"{col}_mean_{window_hours}h" for col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()
    gcs_mean_cols = [f'GCS_{comp}_mean_{window_hours}h' for comp in ['Eye', 'Verbal', 'Motor']]
    
    if all(col in df_pivot.columns for col in gcs_mean_cols):
        df_pivot[f'GCS_Total_mean_{window_hours}h'] = df_pivot[gcs_mean_cols].sum(axis=1, skipna=False)
        # print("Calculated GCS_Total_mean_24h.") # Less verbose
    # print(f"Aggregated {os.path.basename(filepath)} MEAN features shape: {df_pivot.shape}") # Less verbose
    return df_pivot


def preprocess_data(window_hours=24):
    print(f"\n--- Starting Preprocessing for window_hours = {window_hours} ---")
    # --- 1. Load Core Data ---
    print("Loading core data (Admissions, Patients)...")
    try:
        df_adm = pd.read_csv(admissions_file, compression='gzip', parse_dates=['admittime', 'dischtime', 'deathtime'])
        df_pat = pd.read_csv(patients_file, compression='gzip', parse_dates=['dod'])
    except FileNotFoundError as e: print(f"Error: Core file not found: {e}. Exiting."); exit()
    except Exception as e: print(f"Error loading core files: {e}. Exiting."); exit()

    df = pd.merge(df_adm, df_pat, on='subject_id', how='inner')
    print(f"Loaded and merged admissions/patients: {df.shape[0]} rows")

    print("\nCalculating Age at Admission...")
    age_col_name = 'age_at_admission'
    temp_raw_age_col = 'age_at_admission_RAW_temp' # Use temporary name

    try:
        required_cols = ['admittime', 'anchor_age', 'anchor_year'] # Use 'anchor_year' directly
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' is missing for age calculation. Check merge with patients table.")

        print("  Processing admittime...")
        df['admittime'] = pd.to_datetime(df['admittime'], errors='coerce')
        initial_rows = df.shape[0]
        df.dropna(subset=['admittime'], inplace=True)
        rows_dropped = initial_rows - df.shape[0]
        if rows_dropped > 0: print(f"    Dropped {rows_dropped} rows due to invalid 'admittime'.")
        if df.empty: raise ValueError("No valid admittime found, cannot calculate age.")

        df['admission_year_temp'] = df['admittime'].dt.year

        print("  Processing anchor_age...")
        df['anchor_age'] = pd.to_numeric(df['anchor_age'], errors='coerce')
        initial_rows = df.shape[0]
        df.dropna(subset=['anchor_age'], inplace=True)
        rows_dropped = initial_rows - df.shape[0]
        if rows_dropped > 0: print(f"    Dropped {rows_dropped} rows due to invalid 'anchor_age'.")
        if df.empty: raise ValueError("No valid anchor_age found, cannot calculate age.")

        df['anchor_age'] = df['anchor_age'].astype(float)

        print("  Processing anchor_year (from patients table)...")
        # Use temporary name, ensure numeric and integer
        df['anchor_year_temp'] = pd.to_numeric(df['anchor_year'], errors='coerce')
        initial_rows = df.shape[0]
        df.dropna(subset=['anchor_year_temp'], inplace=True) # Drop if anchor_year itself is NaN
        rows_dropped = initial_rows - df.shape[0]
        if rows_dropped > 0: print(f"    Dropped {rows_dropped} rows due to invalid 'anchor_year'.")
        if df.empty: raise ValueError("No valid anchor_year found, cannot calculate age.")
        df['anchor_year_temp'] = df['anchor_year_temp'].astype(int)

        # --- Calculate RAW age using correct (shifted) years ---
        print("  Calculating raw age...")
        df[temp_raw_age_col] = (df['admission_year_temp'] - df['anchor_year_temp']) + df['anchor_age']
        print("  DEBUG: Raw age distribution BEFORE capping/final checks:")
        if temp_raw_age_col in df.columns:
            print(df[temp_raw_age_col].describe())
            print(f"    Count RAW ages > 89: {(df[temp_raw_age_col] > 89).sum()}")
            print(f"    Count RAW ages <= 89: {(df[temp_raw_age_col] <= 89).sum()}")
        else: print("    Could not calculate RAW age.")

        # --- Apply final checks (NaN/negative) on RAW age ---
        initial_rows = df.shape[0]
        invalid_age_mask = df[temp_raw_age_col].isnull() | (df[temp_raw_age_col] < 0)
        invalid_age_count = invalid_age_mask.sum()
        if invalid_age_count > 0:
            print(f"  Warning: Found {invalid_age_count} invalid RAW age results (NaN or < 0). Dropping.")
            df = df[~invalid_age_mask].copy()


        # --- Apply final age cap (>89 -> 90) ---
        print("  Applying final age cap (>89 -> 90)...")
        if temp_raw_age_col not in df.columns: raise ValueError(f"Intermediate raw age column '{temp_raw_age_col}' missing before capping.")
        # Ensure the column exists before applying
        if age_col_name not in df.columns: df[age_col_name] = np.nan # Create column if missing
        df[age_col_name] = df[temp_raw_age_col].apply(lambda x: 90.0 if x > 89 else float(x)) # Ensure result is float


        # --- Clean up intermediate columns ---
        # DO NOT drop the original anchor_year, anchor_age as they might be needed elsewhere if defined in potential_features separately
        df.drop(columns=['admission_year_temp', 'anchor_year_temp', temp_raw_age_col], inplace=True, errors='ignore')

        print(f"\n'{age_col_name}' calculation complete. Final distribution BEFORE merging:")
        if age_col_name in df.columns: print(df[age_col_name].describe())
        else: print(f"'{age_col_name}' column not found!")

    # --- Keep the existing except blocks ---
    except KeyError as ke:
        print(f"CRITICAL ERROR during age calculation: Missing Key {ke}. Age column incorrect.")
        df[age_col_name] = np.nan
    except ValueError as ve:
        print(f"CRITICAL ERROR during age calculation: ValueError {ve}. Age column incorrect.")
        df[age_col_name] = np.nan
    except Exception as e:
        print(f"CRITICAL ERROR during age calculation: {type(e).__name__} - {e}. Age column incorrect.")
        df[age_col_name] = np.nan

    # --- Safety Check ---
    if age_col_name not in df.columns:
        print(f"CRITICAL SAFETY FAIL: Column '{age_col_name}' does not exist after age calculation block!")
        exit() # Exit if age calculation failed critically
    elif df[age_col_name].isnull().all():
         print(f"CRITICAL SAFETY FAIL: Column '{age_col_name}' is all NaN after age calculation block!")
         exit() # Exit if age calculation failed critically
    else:
         print(f"Sanity Check: '{age_col_name}' exists and is not all NaN after calculation block.")

    # --- 2. Aggregate Time-Series Features ---
    print("\nAggregating time-series features...")
    df_labs_agg = load_and_aggregate_mean(filepath=labevents_file, relevant_itemids=lab_itemids, admissions_df=df[['hadm_id', 'admittime']], id_col='hadm_id', window_hours=window_hours)
    df_vitals_agg = load_and_aggregate_mean(filepath=chartevents_file, relevant_itemids=vital_itemids, admissions_df=df[['hadm_id', 'admittime']], id_col='stay_id', window_hours=window_hours)

    if df_labs_agg is not None: df = pd.merge(df, df_labs_agg, on='hadm_id', how='left')
    if df_vitals_agg is not None: df = pd.merge(df, df_vitals_agg, on='hadm_id', how='left')
    print(f"Shape after merging aggregated features: {df.shape}")
    print(f"'{age_col_name}' describe() AFTER merging aggregated features:")
    if age_col_name in df.columns: print(df[age_col_name].describe()) 
    else: print("Age column not found!")

    # --- 3. Add Counts ---
    print("\nAdding counts...")
    # Diagnosis Count
    try:
        df_diag = pd.read_csv(diagnoses_file, compression='gzip', usecols=['hadm_id'])
        diag_counts = df_diag.groupby('hadm_id').size().reset_index(name='diagnosis_count')
        df = pd.merge(df, diag_counts, on='hadm_id', how='left'); df['diagnosis_count'] = df['diagnosis_count'].fillna(0)
    except Exception as e: print(f"Warn: Diag count failed {e}"); df['diagnosis_count'] = 0
    # Prescription Count
    presc_count_col = f'presc_count_{window_hours}h'
    try:
        df_presc = pd.read_csv(prescriptions_file, compression='gzip', usecols=['hadm_id', 'starttime'], parse_dates=['starttime'])
        df_presc = pd.merge(df_presc, df[['hadm_id', 'admittime']], on='hadm_id', how='left').dropna(subset=['admittime', 'starttime'])
        df_presc['time_delta'] = df_presc['starttime'] - df_presc['admittime']
        df_presc_win = df_presc[(df_presc['time_delta'] >= timedelta(0)) & (df_presc['time_delta'] <= timedelta(hours=window_hours))]
        presc_counts = df_presc_win.groupby('hadm_id').size().reset_index(name=presc_count_col)
        df = pd.merge(df, presc_counts, on='hadm_id', how='left'); df[presc_count_col] = df[presc_count_col].fillna(0)
    except Exception as e: print(f"Warn: Presc count failed {e}"); df[presc_count_col] = 0


    # --- 4. Feature Engineering  ---
    print("\Categorising/grouping race...")
    race_map = { 
        'WHITE': 'White', 'WHITE - RUSSIAN': 'White', 'WHITE - OTHER EUROPEAN': 'White',
        'WHITE - BRAZILIAN': 'White', 'WHITE - EASTERN EUROPEAN': 'White', 'PORTUGUESE': 'White',
        'BLACK/AFRICAN AMERICAN': 'Black', 'BLACK/CAPE VERDEAN': 'Black', 'BLACK/HAITIAN': 'Black',
        'BLACK/AFRICAN': 'Black', 'BLACK/CARIBBEAN ISLAND': 'Black',
        'HISPANIC OR LATINO': 'Hispanic', 'HISPANIC/LATINO - PUERTO RICAN': 'Hispanic', 'HISPANIC/LATINO - DOMINICAN': 'Hispanic',
        'HISPANIC/LATINO - GUATEMALAN': 'Hispanic', 'HISPANIC/LATINO - CUBAN': 'Hispanic', 'HISPANIC/LATINO - SALVADORAN': 'Hispanic',
        'HISPANIC/LATINO - CENTRAL AMERICAN': 'Hispanic', 'HISPANIC/LATINO - MEXICAN': 'Hispanic', 'HISPANIC/LATINO - COLOMBIAN': 'Hispanic',
        'HISPANIC/LATINO - HONDURAN': 'Hispanic', 'SOUTH AMERICAN': 'Hispanic', 'ASIAN - SOUTH EAST ASIAN': 'Asian', 'HISPANIC/LATINO - COLUMBIAN': 'Hispanic',
        'ASIAN': 'Asian', 'ASIAN - CHINESE': 'Asian', 'ASIAN - ASIAN INDIAN': 'Asian', 'ASIAN - VIETNAMESE': 'Asian',
        'ASIAN - FILIPINO': 'Asian', 'ASIAN - CAMBODIAN': 'Asian', 'ASIAN - KOREAN': 'Asian', 'ASIAN - JAPANESE': 'Asian',
        'ASIAN - THAI': 'Asian', 'ASIAN - OTHER': 'Asian',
        'OTHER': 'Other/Unknown', 'UNKNOWN': 'Other/Unknown', 'UNABLE TO OBTAIN': 'Other/Unknown',
        'PATIENT DECLINED TO ANSWER': 'Other/Unknown', 'AMERICAN INDIAN/ALASKA NATIVE': 'Other/Unknown',
        'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'Other/Unknown', 'MULTIPLE RACE/ETHNICITY': 'Other/Unknown'
    }
    if 'race' in df.columns:
        df['race_grouped'] = df['race'].replace(race_map)
        original_races = df['race'].unique()
        mapped_races = df['race_grouped'].unique()
        unmapped = [r for r in original_races if r not in race_map and r not in mapped_races]
        if unmapped: print(f"Warn: Unmapped races: {unmapped}. Assigning 'Other/Unknown'.")
        df.loc[df['race'].isna() | df['race_grouped'].isin(unmapped), 'race_grouped'] = 'Other/Unknown'
        df['race'] = df['race_grouped'].fillna('Other/Unknown')
    else: print("Warn: 'race' column not found for grouping.")

    adm_loc_map = {
        'EMERGENCY ROOM': 'Emergency', 'WALK-IN/SELF REFERRAL': 'Emergency',
        'TRANSFER FROM HOSPITAL': 'Transfer In', 'TRANSFER FROM SKILLED NURSING FACILITY': 'Transfer In',
        'AMBULATORY SURGERY TRANSFER': 'Transfer In', 'CLINIC REFERRAL': 'Referral', 'PHYSICIAN REFERRAL': 'Referral',
        'INTERNAL TRANSFER TO OR FROM PSYCH': 'Internal/Procedural', 'PACU': 'Internal/Procedural', 'PROCEDURE SITE': 'Internal/Procedural',
        'INFORMATION NOT AVAILABLE': 'Unknown'
    }
    if 'admission_location' in df.columns:
        df['admission_location_grouped'] = df['admission_location'].replace(adm_loc_map)
        original_adm_loc = df['admission_location'].unique()
        mapped_adm_loc = df['admission_location_grouped'].unique()
        unmapped = [r for r in original_adm_loc if r not in adm_loc_map and r not in mapped_adm_loc]
        if unmapped: print(f"Warn: Unmapped admission locations: {unmapped}. Assigning 'Unknown'.")
        df.loc[df['admission_location'].isna() | df['admission_location_grouped'].isin(unmapped), 'admission_location_grouped'] = 'Unknown'
        df['admission_location'] = df['admission_location_grouped'].fillna('Unknown')
    else: print("Warn: 'admission_location' column not found for grouping.")

    adm_type_map = {
        'DIRECT EMER.': 'Emergency', 'EW EMER.': 'Emergency',
        'URGENT': 'Urgent', 'ELECTIVE': 'Elective', 'SURGICAL SAME DAY ADMISSION': 'Elective',
        'AMBULATORY OBSERVATION': 'Observation', 'DIRECT OBSERVATION': 'Observation', 'EU OBSERVATION': 'Observation', 'OBSERVATION ADMIT': 'Observation'
    }

    if 'admission_type' in df.columns:
        df['adm_type_grouped'] = df['admission_type'].replace(adm_type_map)
        original_adm_types = df['admission_type'].unique()
        mapped_adm_types = df['adm_type_grouped'].unique()
        unmapped = [r for r in original_adm_types if r not in adm_type_map and r not in mapped_adm_types]
        if unmapped: print(f"Warn: Unmapped admission types: {unmapped}. Assigning 'Unknown'.")
        df.loc[df['admission_type'].isna() | df['adm_type_grouped'].isin(unmapped), 'adm_type_grouped'] = 'Unknown'
        df['admission_type'] = df['adm_type_grouped'].fillna('Unknown')
    else: print("Warn: 'admission_type' column not found for grouping.")

    print_value_counts(df, df.columns, "After Loading & Initial Calcs")

    # --- 5. Define Potential Features ---
    target = 'hospital_expire_flag'
    base_features = [
        'admission_type', 'admission_location', 'insurance', 'language', 'marital_status',
        'gender', age_col_name, 'race',
        'diagnosis_count', presc_count_col
    ]
    lab_vital_features = []
    if df_labs_agg is not None: lab_vital_features.extend([col for col in df_labs_agg.columns if col not in ['hadm_id']])
    if df_vitals_agg is not None:
        vitals_cols = [col for col in df_vitals_agg.columns if col not in ['hadm_id']]
        gcs_total_col = f'GCS_Total_mean_{window_hours}h'
        gcs_component_cols = [f'GCS_{comp}_mean_{window_hours}h' for comp in ['Eye', 'Verbal', 'Motor']]
        if gcs_total_col in vitals_cols: vitals_cols = [c for c in vitals_cols if c not in gcs_component_cols] # Keep total, remove components
        lab_vital_features.extend(vitals_cols)

    potential_features = list(set(base_features + lab_vital_features))
    # IMPORTANT: Filter potential_features to only those actually present in df after merges/calcs
    potential_features = [f for f in potential_features if f in df.columns]
    print(f"\nIdentified {len(potential_features)} potential features available in DataFrame.")
    if age_col_name not in potential_features and age_col_name in df.columns:
         print(f"DEBUG: '{age_col_name}' exists in df but not in potential_features list!")
         # Add it back if missing somehow? Should be included via base_features if calculation worked.
         potential_features.append(age_col_name)
    elif age_col_name not in df.columns:
         print(f"DEBUG: '{age_col_name}' does not exist in df columns at this stage!")

    # --- Select columns for df_model ---
    cols_to_keep = ['subject_id', 'hadm_id', target] + potential_features
    df_model = df[cols_to_keep].copy()
    if target not in df_model.columns: raise ValueError(f"Target column '{target}' not found!")
    initial_rows_target = df_model.shape[0]
    df_model.dropna(subset=[target], inplace=True)
    if df_model.shape[0] < initial_rows_target: print(f"Dropped {initial_rows_target - df_model.shape[0]} rows with missing target.")
    df_model[target] = df_model[target].astype(int)
    print(f"\nDEBUG: df_model shape BEFORE Step 6: {df_model.shape}")
    print(f"DEBUG: '{age_col_name}' describe() in df_model BEFORE Step 6:")
    if age_col_name in df_model.columns: print(df_model[age_col_name].describe())
    else: print("Age column not found!")

    # 6a. Impute STATIC Categoricals
    static_categoricals_to_impute = ['admission_type', 'admission_location', 'insurance', 'language', 'marital_status', 'gender', 'race_grouped']
    print("Imputing static categoricals with 'Unknown'...")
    imputed_static_count = 0
    for col in static_categoricals_to_impute:
        if col in df_model.columns:
            if df_model[col].isnull().any():
                imputed_static_count += 1; fill_value = 'Unknown'
                df_model[col] = df_model[col].astype(object).fillna(fill_value)
            df_model[col] = pd.Categorical(df_model[col]) # Ensure categorical
        else: print(f"Warn: Static cat '{col}' not found.")
    if imputed_static_count > 0: print(f"Imputed NaNs in {imputed_static_count} static columns.")
    else: print("No NaNs found in static categoricals.")

    # 6b. Define CRITICAL Features & Drop Rows
    critical_features = [
        age_col_name, 'Heart_Rate_mean_24h', 'MAP_mean_24h', 'Resp_Rate_mean_24h',
        'SpO2_mean_24h', 'Temp_C_mean_24h', 'GCS_Total_mean_24h', 'Lactate_mean_24h', 'Creatinine_mean_24h']
    critical_features = [f for f in critical_features if f in df_model.columns] # Filter list to existing cols
    print(f"\nDEBUG: Critical features selected: {critical_features}")
    if not critical_features: print("WARN: No critical features found/defined!")
    else:
        print(f"Dropping rows based on NaNs in CRITICAL features ({len(critical_features)} features)...")
        initial_rows_crit = df_model.shape[0]
        # --- DEBUG: Check NaNs in critical features BEFORE drop ---
        nans_before_crit_drop = df_model[critical_features].isnull().sum()
        print(f"  NaN counts BEFORE critical drop:\n{nans_before_crit_drop[nans_before_crit_drop > 0]}")
        # --- END DEBUG ---
        df_model.dropna(subset=critical_features, inplace=True)
        rows_after_critical_drop = df_model.shape[0]
        print(f"  Rows remaining after critical drop: {rows_after_critical_drop} ({rows_after_critical_drop*100/max(1, initial_rows_crit):.2f}%)")
        print(f"  DEBUG: Age distribution AFTER critical drop:")
        if age_col_name in df_model.columns: print(df_model[age_col_name].describe())
        else: print("  Age column not found!")

    # 6c. Impute Remaining NUMERICAL Features
    print("\nImputing remaining numerical features with median...")
    all_numerical_cols = df_model.select_dtypes(include=np.number).columns.tolist()
    cols_to_exclude = ['subject_id', 'hadm_id', target] + critical_features
    numerical_cols_to_impute = [f for f in all_numerical_cols if f not in cols_to_exclude]
    if not numerical_cols_to_impute: print("No numerical columns left for imputation.")
    else:
        cols_with_nans = df_model[numerical_cols_to_impute].isnull().any()
        cols_needing_imputation = cols_with_nans[cols_with_nans].index.tolist()
        if cols_needing_imputation:
            print(f"  Numerical columns needing imputation: {len(cols_needing_imputation)}")
            imputer = SimpleImputer(strategy='median')
            df_model[cols_needing_imputation] = imputer.fit_transform(df_model[cols_needing_imputation])
            print(f"  Imputed NaNs using median.")
        else: print("  No NaNs found in remaining numerical columns.")

    # Final check for NaNs
    # Redefine potential_features based on CURRENT df_model columns to avoid errors
    current_potential_features = [f for f in potential_features if f in df_model.columns]
    nans_final = df_model[current_potential_features].isnull().sum().sum()
    print(f"\nFinal check: Total NaNs remaining in potential feature columns: {nans_final}")
    if nans_final > 0: print(f"WARNING: NaNs still present!\n{df_model[current_potential_features].isnull().sum()[df_model[current_potential_features].isnull().sum() > 0]}")

    # --- 7. Encode Categorical Features & Split ---
    print("\nNumerically encoding categorical features and splitting data...")
    categorical_cols = df_model.select_dtypes(include=['category', 'object']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in ['subject_id', 'hadm_id', target]] # Exclude IDs/target

    categorical_mappings = {}
    print(f"Applying numerical encoding to: {list(categorical_cols)}")
    for col in categorical_cols:
        if not pd.api.types.is_categorical_dtype(df_model[col]): df_model[col] = pd.Categorical(df_model[col])
        categorical_mappings[col] = dict(enumerate(df_model[col].cat.categories))
        df_model[col] = df_model[col].cat.codes

    # Define final features list based on columns *currently* in df_model
    final_features = [col for col in df_model.columns if col not in ['subject_id', 'hadm_id', target]]
    print(f"\nDEBUG: Final features ({len(final_features)}) for model: {final_features}")
    if age_col_name not in final_features:
        print(f"CRITICAL DEBUG WARNING: '{age_col_name}' IS NOT in final_features list!")

    # Save mappings
    mapping_file = os.path.join(output_path, f'categorical_mappings_{window_hours}h.pkl') # Unique name per window
    try:
        with open(mapping_file, 'wb') as f: pickle.dump(categorical_mappings, f)
        print(f"Saved categorical mappings to {mapping_file}")
    except Exception as e: print(f"Error saving mappings: {e}")

    # Define X, y, groups
    X = df_model[final_features]
    y = df_model[target]
    groups = df_model['subject_id']

    print(f"\nFinal shapes before split: X={X.shape}, y={y.shape}")
    print(f"Target distribution: {y.value_counts(normalize=True)}")
    print(f"'{age_col_name}' describe() in final X:")
    if age_col_name in X.columns: print(X[age_col_name].describe()) # Describe age in final X
    else: print("Age column not found in final X!")


    if X.shape[0] < 2 or len(np.unique(groups)) < 2:
        print("\nError: Not enough data/groups remaining for split. Skipping modeling/plotting for this window.")
        return None, None, None, None # Return None to indicate failure for this window

    # Save final processed data before split (optional)
    df_model.to_csv(f'{output_path}/mimic_mortality_{window_hours}h_processed.csv', index=False)

    return X, y, groups, categorical_mappings



# --- Control Flag ---
LOAD_PREPROCESSED = False # Set to True to load saved data, False to re-run preprocessing

# --- Main Execution Loop ---
for window_hours in [1,24,48,72]: # Keep focus on 24h for now, expand later
    print(f"\n======= Processing Window: {window_hours} hours =======")

    # Define expected filenames
    processed_data_file = os.path.join(output_path, f'mimic_mortality_{window_hours}h_processed.csv')
    mapping_file = os.path.join(output_path, f'categorical_mappings_{window_hours}h.pkl')
    target = 'hospital_expire_flag'
    group_col = 'subject_id'
    id_cols = ['subject_id', 'hadm_id']

    data_loaded_successfully = False
    if LOAD_PREPROCESSED and os.path.exists(processed_data_file) and os.path.exists(mapping_file):
        print(f"Attempting to load preprocessed data from {output_path}...")
        try:
            df_loaded = pd.read_csv(processed_data_file)
            with open(mapping_file, 'rb') as f:
                categorical_mappings = pickle.load(f)

            # --- Reconstruct X, y, groups ---
            # Ensure target and group column exist
            if target not in df_loaded.columns:
                raise ValueError(f"Target column '{target}' not found in loaded file.")
            if group_col not in df_loaded.columns:
                 raise ValueError(f"Group column '{group_col}' not found in loaded file.")

            y = df_loaded[target]
            groups = df_loaded[group_col]
            # X contains all columns except target and IDs
            feature_cols = [col for col in df_loaded.columns if col not in [target] + id_cols]
            X = df_loaded[feature_cols]


            for col in categorical_mappings.keys():
                 if col in X.columns:
                      X[col] = X[col].astype(int)


            print(f"Successfully loaded and reconstructed data. X shape: {X.shape}")
            data_loaded_successfully = True

        except Exception as e:
            print(f"Error loading preprocessed data: {e}. Running full preprocessing...")
            data_loaded_successfully = False # Ensure flag is reset

    if not data_loaded_successfully:
        print("\nRunning full preprocessing...")
        # Run the full preprocessing function
        processed_data_tuple = preprocess_data(window_hours=window_hours)
        if processed_data_tuple[0] is None: # Check if preprocessing failed
            print(f"Preprocessing failed for window {window_hours}h. Skipping.")
            continue # Skip to the next window_hours value
        X, y, groups, categorical_mappings = processed_data_tuple
        # Data is saved inside preprocess_data now

    # --- Proceed with Split, Train, Explain, Plot (using loaded or processed X, y, groups, mappings) ---

    if X is None or X.empty:
        print(f"No data available for window {window_hours}h after loading/processing. Skipping.")
        continue

    print(f"\nSplitting data for window {window_hours}h...")
    # Ensure enough data for split
    if X.shape[0] < 5 or len(np.unique(groups)) < 2: # Increased minimum slightly
         print(f"Warning: Not enough data or groups for reliable split ({X.shape[0]} rows, {len(np.unique(groups))} groups). Skipping window {window_hours}h.")
         continue

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    try:
        train_idx, test_idx = next(gss.split(X, y, groups))
    except ValueError as e:
        print(f"Error during GroupShuffleSplit (likely too few groups or samples per group): {e}. Skipping window {window_hours}h.")
        continue

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Check if splits are empty
    if X_train.empty or X_test.empty:
        print(f"Warning: Train or test split resulted in empty data. Skipping window {window_hours}h.")
        continue

    print(f"\n--- Completed window_hours = {window_hours} ---")

print("\nAll processing and plotting finished.")