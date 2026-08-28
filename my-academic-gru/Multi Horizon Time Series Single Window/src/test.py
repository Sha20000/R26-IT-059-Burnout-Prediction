import pandas as pd
import numpy as np

student_info = pd.read_csv('../data/raw/studentInfo.csv')

# TRUE concurrent check: same
# id_student AND same
# code_presentation, but different
# code_module
concurrent_counts = student_info.groupby(
    ['id_student', 'code_presentation']
)['code_module'].apply(
    lambda x: len(x.unique())
)



truly_concurrent = concurrent_counts[
    concurrent_counts > 1
]

print(f"Student-semester combos with "
      f"2+ CONCURRENT modules: "
      f"{len(truly_concurrent)}")

print(f"Unique students involved in "
      f"concurrent enrollment: "
      f"{truly_concurrent.index.get_level_values('id_student').nunique()}")

print(f"Max concurrent modules in "
      f"one semester: "
      f"{truly_concurrent.max()}")

# Also show breakdown
print("\nDistribution of concurrent module counts:")
print(truly_concurrent.value_counts().sort_index())


#Inspect one actual concurrent student

concurrent_index = truly_concurrent.index
sample_student_id, sample_presentation = concurrent_index[0]

print(f"\n\nInspecting student:{sample_student_id}")
print(f"Presentation: {sample_presentation}")

print("\nTheir modules:")
print(student_info[(student_info['id_student'] == sample_student_id) & 
                   (student_info['code_presentation'] == sample_presentation)][['code_module','final_result']])


#Check current assessment

student_ass = pd.read_csv("../data/raw/studentAssessment.csv")
assessments = pd.read_csv("../data/raw/assessments.csv")

#Merge like your main pipeline does

ass = student_ass.merge(
    assessments[['id_assessment','code_module',
                 'code_presentation','date']],
                 on='id_assessment',how='left'
)

#Look at this student's raw records
print("\n\nTheir RAW assessment records:")
print(ass[ass['id_student'] == sample_student_id][['id_student','code_module','date_submitted','score']
                                                  ].sort_values('date_submitted'))

#See what your current ass_weekly code would produce for this student
ass['week'] = (ass['date_submitted']/7).astype(int)+1


#Group exactly like your current ass_weekly code does

ass_weekly_current = ass.groupby(
    ['id_student', 'week'], as_index=False
).agg(
    F3_avg_score=('score','mean'),
    F4_num_submitted=('score','count')
)

print("\n\nCurrent (BLENDED) weekly output for this student:")
print(ass_weekly_current[ass_weekly_current['id_student'] == sample_student_id])

#Fix

ass_weekly_fixed = ass.groupby(
    ['id_student','code_module','week'],
      as_index=False).agg(
    F3_avg_score=('score','mean'),
    F4_num_submitted=('score','count')
)

print("\n\nFIXED (SEPARATED) weekly output for this student:")
print(ass_weekly_fixed[
    ass_weekly_fixed['id_student'] == sample_student_id
])

#Step 6- VLE Data 

student_vle = pd.read_csv("../data/raw/studentVle.csv")

vle = student_vle.copy()

vle['week'] = (vle['date']/7).astype(int)+1
N_WEEKS = 17
vle = vle[vle['week'] <= N_WEEKS]

vle_weekly = vle.groupby(['id_student','code_module','code_presentation','week'], as_index=False).agg(
    F1_login_count = ('sum_click','count'),
    F2_vle_clicks = ('sum_click','sum')
)

print(f"\nVLE weekly groups: {len(vle_weekly)}")

#Step 7: Full assessment feature building, with code_module fix

ass_full = student_ass.merge(
    assessments[['id_assessment', 'code_module', 'code_presentation', 'date','assessment_type','weight']],
    on='id_assessment', how='left'
)

ass_full = ass_full.dropna(subset=['date'])
ass_full['week'] = (ass_full['date']/7).astype(int)+1
ass_full['score'] = ass_full['score'].fillna(0)
ass_full['on_time'] = (ass_full['date_submitted'] <= ass_full['date']).astype(int)
ass_full['is_late'] = (ass_full['date_submitted'] > ass_full['date']).astype(int)
ass_full = ass_full[ass_full['week'] <= N_WEEKS]


#code module fix: group by code_module as well
ass_weekly_full = ass_full.groupby(
    ['id_student','code_module','code_presentation','week'], as_index=False
).agg(
    F3_avg_score=('score','mean'),
    F4_num_submitted=('score','count'),
    F5_on_time_rate=('on_time','sum'),
    days_with_activity = ('date_submitted','nunique'),
    F9_score_variance=('score','std'),
    F10_min_score=('score','min'),
    F11_max_score=('score','max'),
    F12_late_count=('is_late','sum')


)

ass_weekly_full['F6_inactive_days'] = (7- ass_weekly_full['days_with_activity']).clip(lower=0)
ass_weekly_full['F9_score_variance'] = ass_weekly_full['F9_score_variance'].fillna(0)
ass_weekly_full = ass_weekly_full.drop(columns=['days_with_activity'])

print(f"\nFull assessment weekly groups: {len(ass_weekly_full):,}")


# ALL 8 assessment features, correctly split by module
print("\nFull feature check for student 29820:")
print(ass_weekly_full[ass_weekly_full['id_student'] == 29820])

#Filter only concurrent students for final output

concurrent_student_ids = truly_concurrent.index.get_level_values('id_student').unique()

print(f"\nTotal concurrent students: {len(concurrent_student_ids)}")

ass_weekly_concurrent = ass_weekly_full[
    ass_weekly_full['id_student'].isin(concurrent_student_ids)
]

print(f"Assessment weekly rows for concurrent students: "
      f"{len(ass_weekly_concurrent):,}")

print(f"Unique students in filtered result: "
      f"{ass_weekly_concurrent['id_student'].nunique():,}")

vle_weekly_concurrent = vle_weekly[
    vle_weekly['id_student'].isin(concurrent_student_ids)
]

print(f"VLE weekly rows for concurrent students: "
      f"{len(vle_weekly_concurrent):,}")

#Investigate missing students

students_with_assessments = set(
    ass_weekly_concurrent['id_student'].unique()
) 

all_concurrent = set(concurrent_student_ids)

missing_students = all_concurrent - students_with_assessments
print(f"\nConcurrent students with NO assessment records:" 
      f"{len(missing_students)}")

if missing_students:
    sample_missing = list(missing_students)[0]
    print(f"\nExample missing student: {sample_missing}")
    print("\nTheir RAW assessment records(before ANY filtering):")
    print(student_ass[student_ass['id_student']== sample_missing])


#Build the per module grid for concurrent students

#Only keep concurrent students who have assessment activities
valid_concurrent_ids = ass_weekly_concurrent['id_student'].unique()

#Build the base "student x modules table"

student_info['label'] = student_info[
    'final_result'].apply(
    lambda x: 0 if x in ['Pass','Distinction'] else 1)

concurrent_student_modules = student_info[(student_info['id_student'].isin(valid_concurrent_ids))][['id_student','code_module','code_presentation','label']].drop_duplicates()

print(f"\nStudent-module combinations to build: "
      f"{len(concurrent_student_modules)}")


#Cross with all 17 weeks
all_weeks = pd.DataFrame({'week': range(1, N_WEEKS+1)})
grid_concurrent = concurrent_student_modules.merge(all_weeks, how='cross')

print(f"Grid size (before merging features): "
      f"{len(grid_concurrent):,} rows")

#Merge in assessment features 

grid_concurrent = grid_concurrent.merge(
    ass_weekly_concurrent[
        ['id_student','code_module','week',
        'F3_avg_score', 'F4_num_submitted',
        'F5_on_time_rate','F6_inactive_days',
        'F9_score_variance','F10_min_score',
        'F11_max_score','F12_late_count']
    ], 
    on=['id_student','code_module','week'], how='left'
)

#Merge in Vle features

grid_concurrent = grid_concurrent.merge(

    vle_weekly_concurrent[[
        'id_student','code_module','code_presentation',
        'week','F1_login_count','F2_vle_clicks'
    ]],
    on=['id_student','code_module','code_presentation','week'], how='left'

)

grid_concurrent = grid_concurrent.sort_values(['id_student','code_module','week']).reset_index(drop=True)

print(f"Grid after merging features: {len(grid_concurrent):,} rows")

#Verfiy on student 29820 again

print("\n Check for student 29820 : ")

print(grid_concurrent[grid_concurrent['id_student'] == 29820][['id_student','code_module','week','F3_avg_score','F1_login_count']].
      head(10))

#Fill missing values 

grid_concurrent['F1_login_count'] = grid_concurrent['F1_login_count'].fillna(0)
grid_concurrent['F2_vle_clicks'] = grid_concurrent['F2_vle_clicks'].fillna(0)

grid_concurrent['F3_avg_score'] = grid_concurrent.groupby(
    ['id_student','code_module'])['F3_avg_score'].transform(
        lambda x:x.ffill().fillna(50.0)
    )

grid_concurrent['F4_num_submitted'] = grid_concurrent['F4_num_submitted'].fillna(0)
grid_concurrent['F5_on_time_rate'] = grid_concurrent['F5_on_time_rate'].fillna(0)
grid_concurrent['F6_inactive_days'] = grid_concurrent['F6_inactive_days'].fillna(7.0)
grid_concurrent['F9_score_variance'] = grid_concurrent['F9_score_variance'].fillna(0)

grid_concurrent['F10_min_score'] = grid_concurrent.groupby(
    ['id_student','code_module'])['F10_min_score'].transform(
        lambda x:x.ffill().fillna(50.0)
    )

grid_concurrent['F11_max_score'] = grid_concurrent.groupby(
    ['id_student','code_module'])['F11_max_score'].transform(
        lambda x:x.ffill().fillna(50.0)
    )

grid_concurrent['F12_late_count'] = grid_concurrent['F12_late_count'].fillna(0)

print("\nMissing value filled")
print("\nCheck for student 29820 (after filling):")

print(grid_concurrent[grid_concurrent['id_student'] == 29820][['id_student','code_module','week','F3_avg_score']].head(10))



#Calculate F7,F8,F13

#F7 - score change from last week

grid_concurrent['F7_score_change'] = grid_concurrent.groupby(
    ['id_student','code_module'])['F3_avg_score'].diff().fillna(0)

#F8 - running average score

grid_concurrent['F8_running_avg_score'] = grid_concurrent.groupby(
    ['id_student','code_module'])['F3_avg_score'].transform(
        lambda x: x.expanding().mean()
    )

#F13 - weeks since last submission

grid_concurrent['last_active_week'] = grid_concurrent.groupby(
    ['id_student','code_module']).apply(
        lambda g: g['week'].where(
            g['F4_num_submitted']>0
        ).ffill()
    ).reset_index(level=[0,1], drop=True)

grid_concurrent['F13_weeks_since_active'] = (
    grid_concurrent['week'] - grid_concurrent['last_active_week']
).fillna(0).clip(lower=0)

grid_concurrent = grid_concurrent.drop(columns=['last_active_week'])

print("\nF7 , F8 , F13 calculated")

#Verify students for 29820

print("\nCheck for student 29820 (F7, F8, F13):")

print(grid_concurrent[grid_concurrent['id_student'] == 29820][['code_module','week','F3_avg_score',
                                                               'F7_score_change','F8_running_avg_score','F13_weeks_since_active']].head(10))




#Build final X array

FEATURE_NAMES = [
    'F1_login_count', 'F2_vle_clicks', 'F3_avg_score',
    'F4_num_submitted', 'F5_on_time_rate', 'F6_inactive_days',
    'F7_score_change', 'F8_running_avg_score', 'F9_score_variance',
    'F10_min_score', 'F11_max_score', 'F12_late_count',
    'F13_weeks_since_active'
]


#Keep only student-module combos with all 17 weeks

grid_concurrent['student_module_id'] = (
    grid_concurrent['id_student'].astype(str) + '_' +
    grid_concurrent['code_module']
)




week_count = grid_concurrent.groupby('student_module_id')['week'].count()
full_combos = week_count[week_count == N_WEEKS].index
grid_concurrent_final = grid_concurrent[
    grid_concurrent['student_module_id'].isin(full_combos) 
].sort_values(['student_module_id','week'])

n_combos = len(full_combos)
n_features = len(FEATURE_NAMES)

print(f"\nStudent-module combos with all {N_WEEKS} weeks: {n_combos}")

#Build X-shape (combos, 17, 13)

X_concurrent = grid_concurrent_final[FEATURE_NAMES].values.reshape(
    n_combos, N_WEEKS, n_features
).astype(np.float32)

X_concurrent = np.nan_to_num(X_concurrent, nan=0.0, posinf=0.0, neginf=0.0)

print(f"X_concurrent shape: {X_concurrent.shape}")

#Normalization 

from data_loader import get_data

data = get_data(verbose=False)
scaler = data['scaler']

n_combos_check,n_weeks_check,n_features_check = X_concurrent.shape
X_concurrent_flat = X_concurrent.reshape(n_combos_check, -1)
X_concurrent_scaled_flat = scaler.transform(X_concurrent_flat)
X_concurrent = X_concurrent_scaled_flat.reshape(
    n_combos_check, n_weeks_check, n_features_check
).astype(np.float32)


print(f"X_concurrent normalized. Shape: {X_concurrent.shape}")

print("\n=== SANITY CHECK: Normalized feature ranges ===")
print(f"X_concurrent min: {X_concurrent.min():.2f}")
print(f"X_concurrent max: {X_concurrent.max():.2f}")
print(f"X_concurrent mean: {X_concurrent.mean():.2f}")
print(f"X_concurrent std: {X_concurrent.std():.2f}")

# Compare to what training data looked like
print(f"\nTraining X_tr min: {data['X_tr'].min():.2f}")
print(f"Training X_tr max: {data['X_tr'].max():.2f}")








#Keep a lookup
combo_lookup = grid_concurrent_final.groupby(
    'student_module_id', sort=False
)[['id_student','code_module','label']].first()

combo_lookup = combo_lookup.loc[full_combos]

print(f"\nSample lookup entries: ")
print(combo_lookup.head(5))

np.save("../data/processed/X_concurrent.npy", X_concurrent)
combo_lookup.to_csv("../data/processed/combo_lookup.csv")
print("\nSaved X_concurrent.npy and concurrent_lookup.csv")

#Load the model and run inference

import torch
from model_8_unified_gru import UnifiedMultiHorizonGRU

model = UnifiedMultiHorizonGRU(n_features=13)
model.load_state_dict(
    torch.load("../models/saved/model8_unified_gru_best.pt",map_location="cpu")

)
model.eval()

X_tensor = torch.FloatTensor(X_concurrent)

with torch.no_grad():

    predictions_list,_ = model(X_tensor)
    week17_logits = predictions_list[3].squeeze()
    week17_probs = torch.sigmoid(week17_logits).numpy()


WEEK17_THRESHOLD = 0.55
week17_preds = (week17_probs >= WEEK17_THRESHOLD).astype(int)

print(f"\nPrediction generated for {len(week17_preds)} student-module combos")
print(f"Flagged at-risk: {week17_preds.sum()}({100*week17_preds.mean():.1f}%)")

#Attach predictions to the lookup table

combo_lookup_final = combo_lookup.loc[full_combos].copy()
combo_lookup_final['risk_prob'] = week17_probs
combo_lookup_final['prediction'] = week17_preds

print("\nSample predictions: ")
print(combo_lookup_final.head(10))

combo_lookup_final.to_csv("../data/processed/concurrent_predictions.csv")
print("\nSaved concurrent_predictions.csv")


#Recalibrate threshold using concurrent

print("\nRisk probability distribution for concurrent students:")
print(combo_lookup_final['risk_prob'].describe())


#Percentile based threshold
percentile_threshold = combo_lookup_final['risk_prob'].quantile(0.69)
print(f"\n69th percentile threshold: {percentile_threshold:.4f}")

combo_lookup_final['prediction_recalibrated'] = (combo_lookup_final['risk_prob'] >= percentile_threshold).astype(int)

print(f"\nFlagged at-risk (recalibrated): "
      f"{combo_lookup_final['prediction_recalibrated'].sum()} "
      f"({100*combo_lookup_final['prediction_recalibrated'].mean():.1f}%)")

#Check does this correlate at all with actual label

from sklearn.metrics import f1_score

f1 = f1_score(combo_lookup_final['label'], combo_lookup_final['prediction_recalibrated'])
print(f"F1 score with recaliberated threshold: {f1:.4f}")

#Count based threshold vs Max-Pooling

from sklearn.metrics import f1_score,precision_score,recall_score

def show_metrics(name, y_true, y_pred):
    print(f"\n{name}")
    print(f" F1: {f1_score(y_true,y_pred):.4f}")
    print(f" Precision: {precision_score(y_true,y_pred):.4f}")
    print(f" Recall: {recall_score(y_true,y_pred):.4f}")
    print(f" Flagged at-risk: {sum(y_pred)/len(y_pred)}"
          f"({100*sum(y_pred)/len(y_pred):.1f}%)")



results = []
for student_id, group in combo_lookup_final.groupby('id_student'):

    module_preds = group['prediction_recalibrated'].tolist()
    actual = group['label'].max()

    count_based = int(sum(module_preds) > len(module_preds)/2)
    max_pool = int(any(module_preds))

    results.append({
        'id_student': student_id,
        'n_modules': len(module_preds),
        'actual': actual,
        'count_based_pred': count_based,
        'max_pool_pred': max_pool
    })


results_df = pd.DataFrame(results)

print(f"\nTotal students evaluated: {len(results_df)}")
show_metrics("Count-Based Threshold", results_df['actual'], results_df['count_based_pred'])
show_metrics("Max-Pooling", results_df['actual'], results_df['max_pool_pred'])

#Save final results for PP2 slides

results_df.to_csv('../data/processed/pp2_final_results.csv',index=False)
combo_lookup_final.to_csv('../data/processed/pp2_per_module_predictions.csv')


print("\n" + "="*50)
print("PP2 ANALYSIS COMPLETE")
print("="*50)
print("Saved:")
print("  pp2_final_results.csv          (per-student, both approaches)")
print("  pp2_per_module_predictions.csv (per-module, recalibrated)")














