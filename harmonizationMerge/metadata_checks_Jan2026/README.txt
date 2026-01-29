This folder is checking that the harmonization is well done. Especially in the CERAD category. 

1. [GETOBSANDSAVE.PY -> OBS_COLUMN_NAMES.TXT] get obs columns and then save into text file

2. [CORE_COLUMN_NAMES.PY] record "CORE" list  list of vars to be kept

3. [UNIQUEVALUESOBS.PY -> OBS_SELECTED_COLUMNS_UNIQUE_SUMMARY.PY] 
    3.1. Create a unique value script. For each dataset, save in .txt (?) or .csv, unique values for all data types: Sex, Chemistry, batch, projid, individualID, age_death, Braak, CERAD, cogdx, mmse, APOE, Cognitive Status, Donor ID, Last MMSE Score, PMI, Years of educ? cts_mmse_30lv -> last visit
    3.2. Check missing values columns. Check ("CORE") columns & check if there are any missing values prior to merging.
    3.3 Make sure indivID are unique. Ensure they are unique within each dataset.

4. For each var: Check in lit / Rush / Sea-AD to see how the results/metadata were saved & check the definitions of each var to make sure it matches. Note: Create definition doc with this info.

5. For harmonization code:
A) Make sure that you include a "raw" column for: age, sex, pmi, braak, cerad, mmse, apoe, pmi, cogdx.
B) Make sure code only accounts for cases related to current date.
NOTE: 1. Check that values in Fujita + MIT are the same (obs). 2. Only switch SEA-AD labels.