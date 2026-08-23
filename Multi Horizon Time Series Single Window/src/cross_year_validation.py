#Step 1 - Check presentation distribution
#Confirms enough students exist per year to 
#split meaningfully into train/test split

import pandas as pd
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,roc_auc_score,precision_score,recall_score,accuracy_score

student_info = pd.read_csv('../data/raw/studentInfo.csv')

student_info['label'] = student_info['final_result'].apply(
    lambda x: 0 if x in ['Pass', 'Distinction'] else 1
)

print("Student per presentation: ")
print(student_info['code_presentation'].value_counts())

print("\nAt-risk rate per presentation :")
print(student_info.groupby('code_presentation')['label'].mean())

#Step 2 - Link year presentation to 
#saved X_sequences.npy and y_labels.npy files

N_WEEKS = 17

#Reload assessment data

student_ass = pd.read_csv('../data/raw/studentAssessment.csv')
assessments = pd.read_csv('../data/raw/assessments.csv')


ass = student_ass.merge(
    assessments[['id_assessment', 'date', 
                 'assessment_type', 'weight']],
                 on='id_assessment',how='left'
)

ass = ass.dropna(subset=['date'])
ass['week'] = (ass['date']/7).astype(int) + 1
ass['score'] = ass['score'].fillna(0)
ass = ass[ass['week'] <= N_WEEKS]

#Valid Ids 
valid_ids = ass[
    ass.groupby('id_student')['score'].transform('count') > 0
].groupby('id_student')['week'].nunique()
valid_ids = valid_ids[valid_ids >= 2].index

students = student_info[student_info['id_student'].isin(valid_ids)][['id_student','code_presentation']].drop_duplicates(subset='id_student')

print(f"Students matching original filter: {len(students)}")
print(students.head())

#Step 3 - Build complete feature sequences,
#seperately for 2013 students(Train) and 2014 students(Test)

student_vle = pd.read_csv('../data/raw/studentVle.csv')
vle = student_vle.copy()
vle['week'] = (vle['date'] / 7).astype(int) + 1
vle = vle[vle['week'] <= N_WEEKS]

vle_weekly = vle.groupby(
    ['id_student','code_module','code_presentation','week'],
    as_index=False
).agg(
    F1_login_count=('sum_click','count'),
    F2_vle_clicks=('sum_click','sum')
)

ass['on_time'] = (ass['date_submitted']<= ass['date']).astype(int)
ass['is_late'] = (ass['date_submitted']>ass['date']).astype(int)

ass_weekly = ass.groupby(
    ['id_student','week'], as_index=False
).agg(
    F3_avg_score = ('score','mean'),
    F4_num_submitted = ('id_assessment','count'),
    F5_on_time_rate = ('on_time','mean'),
    days_with_activity = ('date_submitted','nunique'),
    F9_score_variance = ('score','std'),
    F10_min_score = ('score','min'),
    F11_max_score = ('score','max'),
    F12_late_count = ('is_late','sum')

)

ass_weekly['F6_inactive_days'] = (7-ass_weekly['days_with_activity']).clip(lower=0)
ass_weekly['F9_score_variance'] = ass_weekly['F9_score_variance'].fillna(0)
ass_weekly = ass_weekly.drop(columns=['days_with_activity'])

print(f"VLE weekly rows: {len(vle_weekly):,}")
print(f"Assessment weekly rows: {len(ass_weekly):,}")

#STEP4: Build the full student-week grid 
#spill into train(2013) and 
#test(2014) sets

# Build the valid student list the SAME way 
# original notebook does (with correct multi-stage filter)

valid_ids_full = ass_weekly[
    ass_weekly['F4_num_submitted'] > 0
].groupby('id_student')['week'].nunique()
valid_ids_full = valid_ids_full[valid_ids_full >= 2].index

students_full = student_info[
    student_info['id_student'].isin(valid_ids)
][['id_student','code_module','code_presentation','label']].drop_duplicates()

print(f"Valid students (matching original notebook logic): {len(students_full):,}")

#SPILL BY YEAR
train_presentation = ['2013B','2013J']
test_presentation = ['2014B','2014J']

students_train = students_full[students_full['code_presentation'].isin(train_presentation)]
students_test = students_full[students_full['code_presentation'].isin(test_presentation)]

print(f"\nTrain students (2013): {len(students_train):,}")
print(f"Test students (2014):  {len(students_test):,}")

# STEP 5: Build the complete 17-week grid
# with all 13 features, then apply the FINAL
# strict filter (exactly 17 weeks)

FEATURE_NAMES = [
    'F1_login_count', 'F2_vle_clicks', 'F3_avg_score',
    'F4_num_submitted', 'F5_on_time_rate', 'F6_inactive_days',
    'F7_score_change', 'F8_running_avg', 'F9_score_variance',
    'F10_min_score', 'F11_max_score', 'F12_late_count',
    'F13_weeks_since_active'
]

all_weeks = pd.DataFrame({'week': range(1, N_WEEKS + 1)})
grid = students_full.merge(all_weeks, how='cross')

grid = grid.merge(
    ass_weekly[[
        'id_student', 'week',
        'F3_avg_score', 'F4_num_submitted',
        'F5_on_time_rate', 'F6_inactive_days',
        'F9_score_variance', 'F10_min_score',
        'F11_max_score', 'F12_late_count']],
    on=['id_student', 'week'], how='left')

grid = grid.merge(
    vle_weekly[[
        'id_student', 'code_module', 'code_presentation',
        'week', 'F1_login_count', 'F2_vle_clicks']],
    on=['id_student', 'code_module', 'code_presentation', 'week'],
    how='left')


grid = grid.sort_values(['id_student', 'week']).reset_index(drop=True)

print(f"Grid built: {len(grid):,} rows")

# Fill missing values (same as original notebook)
grid['F1_login_count'] = grid['F1_login_count'].fillna(0)
grid['F2_vle_clicks'] = grid['F2_vle_clicks'].fillna(0)
grid['F3_avg_score'] = grid.groupby('id_student')['F3_avg_score'].transform(
    lambda x: x.ffill().fillna(50.0))
grid['F4_num_submitted'] = grid['F4_num_submitted'].fillna(0)
grid['F5_on_time_rate'] = grid['F5_on_time_rate'].fillna(1.0)
grid['F6_inactive_days'] = grid['F6_inactive_days'].fillna(7.0)
grid['F9_score_variance'] = grid['F9_score_variance'].fillna(0)
grid['F10_min_score'] = grid.groupby('id_student')['F10_min_score'].transform(
    lambda x: x.ffill().fillna(50.0))
grid['F11_max_score'] = grid.groupby('id_student')['F11_max_score'].transform(
    lambda x: x.ffill().fillna(50.0))
grid['F12_late_count'] = grid['F12_late_count'].fillna(0)

# F7, F8, F13
grid['F7_score_change'] = grid.groupby('id_student')['F3_avg_score'].diff().fillna(0)
grid['F8_running_avg'] = grid.groupby('id_student')['F3_avg_score'].transform(
    lambda x: x.expanding().mean())
grid['last_active_week'] = grid.groupby('id_student').apply(
    lambda g: g['week'].where(g['F4_num_submitted'] > 0).ffill()
).reset_index(level=0, drop=True)
grid['F13_weeks_since_active'] = (
    grid['week'] - grid['last_active_week']
).fillna(0).clip(lower=0)
grid = grid.drop(columns=['last_active_week'])

# FINAL strict filter: exactly 17 weeks
week_counts = grid.groupby('id_student')['week'].count()
full_students = week_counts[week_counts == N_WEEKS].index
grid = grid[grid['id_student'].isin(full_students)]

n_students = len(full_students)
print(f"\nStudents with complete {N_WEEKS} weeks: {n_students:,}")


#Step 6: Build final X,y arrays - split by 
#presentation year into Train (2013) and 
#Test(2014)

n_features = len(FEATURE_NAMES)

#Build X and y for ALL 16,950 students first

X_all = grid[FEATURE_NAMES].values.reshape(
    n_students, N_WEEKS, n_features
).astype(np.float32)

X_all = np.nan_to_num(X_all, nan=0.0,posinf=100.0,neginf=0.0)

y_all = grid.groupby('id_student')['label'].first().values.astype(np.float32)

#Get presentation year for each student

presentation_lookup = grid.groupby('id_student')['code_presentation'].first().loc[full_students]

print(f"X_all_shape: {X_all.shape}")
print(f"y_all_shape: {y_all.shape}")

#Create train/test masks based on presentation year
train_mask = presentation_lookup.isin(train_presentation).values
test_mask = presentation_lookup.isin(test_presentation).values

X_train_year = X_all[train_mask]
y_train_year = y_all[train_mask]    
X_test_year = X_all[test_mask]
y_test_year = y_all[test_mask]

print(f"\nCross-year TRAIN set(2013): {X_train_year.shape}, at-risk rate: {y_train_year.mean()*100:.1f}%")
print(f"Cross-year TEST set(2014):  {X_test_year.shape}, at-risk rate: {y_test_year.mean()*100:.1f}%")

#Save these for next step training
np.save('../data/processed/X_train_2013.npy', X_train_year)
np.save('../data/processed/y_train_2013.npy', y_train_year)
np.save('../data/processed/X_test_2014.npy', X_test_year)
np.save('../data/processed/y_test_2014.npy', y_test_year)

print("\nSaved: X_train_2013.npy, y_train_2013.npy, X_test_2014.npy, y_test_2014.npy")


#Model Architecture 

HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3
HORIZON_INDICES = [3,7,11,16]


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__
        self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(self, gru_output):
        scores = self.attention_weights(gru_output)
        weights = torch.softmax(scores, dim=1)
        attended = (weights*gru_output).sum(dim=1)
        return attended,weights.squeeze(-1)

class PredictionHead(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size,32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32,1)
        )        

    def forward(self,x):
        return self.head(x)


class UnifiedMultiHorizonGRU(nn.Module):

    def __init__(self, n_features, hidden=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS,dropout=DROPOUT,horizon_indices=HORIZON_INDICES):
        super().__init__()
        self.horizon_indices = horizon_indices
        self.n_horizons = len(horizon_indices)
        self.gru = nn.GRU(input_size=n_features,
                          hidden_size=hidden,num_layers=num_layers,
                          batch_first=True,dropout=dropout)
        self.attention = AttentionLayer(hidden)
        self.heads = nn.ModuleList([
            PredictionHead(hidden,dropout) for _ in range(self.n_horizons)
        ])
        self.dropout = nn.Dropout(dropout)


    def forward(self,x):
        gru_out,_ = self.gru(x)
        attended, attn_weights = self.attention(gru_out)
        predictions = []
        for head,idx in zip(self.heads, self.horizon_indices):
            week_state = gru_out[:, idx,:]
            week_state = self.dropout(week_state)
            pred = head(week_state)
            predictions.append(pred)

        return predictions, attn_weights



#Step 7: Train a new model on 2013 data only,
#then evaluate on unseen 2014 data

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

MODELS_PATH = '../models/saved/'
EPOCHS = 500
PATIENCE = 50
BATCH_SIZE = 64
LR = 0.0001

if torch.cuds.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')    

X_train = np.load('../data/processed/X_train_2013.npy')
y_train = np.load('../data/processed/y_train_2013.npy')
X_test  = np.load('../data/processed/X_test_2014.npy')
y_test  = np.load('../data/processed/y_test_2014.npy')

n_train = len(X_train)
idx = np.random.permutation(n_train)
val_size = int(n_train*0.15)
val_idx, tr_idx = idx[:val_size], idx[val_size:]

X_tr_raw, y_tr = X_train[tr_idx], y_train[tr_idx]
X_val_raw, y_val = X_train[val_idx], y_train[val_idx]


scaler_cy = StandardScaler()
X_tr_flat = X_tr_raw.reshape(len(X_tr_raw),-1)
X_tr_scaled = scaler_cy.fit_transform(X_tr_flat).reshape(X_tr_raw.shape).astype(np.float32)

X_val_flat = X_val_raw.reshape(len(X_val_raw),-1)
X_val_scaled = scaler_cy.fit_transform(X_val_flat).reshape(X_val_raw.shape).astype(np.float32)

X_test_flat = X_test.reshape(len(X_test),-1)
X_test_scaled = scaler_cy.fit_transform(X_test_flat).reshape(X_test.shape).astype(np.float32)


print(f"Train: {X_tr_scaled.shape}, Val: {X_val_scaled.shape}, Test(2014): {X_test_scaled.shape}")


X_tr_t = torch.FloatTensor(X_tr_scaled).to(device)
y_tr_t = torch.FloatTensor(y_tr).to(device)
X_val_t = torch.FloatTensor(X_val_scaled).to(device)
y_val_t = torch.FloatTensor(y_val).to(device)
X_test_t = torch.FloatTensor(X_test_scaled).to(device)
y_test_t = torch.FloatTensor(y_test).to(device)

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)

model_cy = UnifiedMultiHorizonGRU(n_features=13).to(device)
optimizer = torch.optim.Adam(model_cy.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

pos_weight_val = (1-y_tr.mean()) / y_tr.mean()
pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

print(f"Class weight : {pos_weight_val:.2f}x")

best_val_f1 = 0
best_epoch = 0
no_improve = 0

print(f"\nTraining cross-year model (2013 -> 2014)...")

for epoch in range(EPOCHS):
      
    model_cy.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions, _ = model_cy(X_batch)
        total_loss = sum(criterion(p.squeeze(-1),y_batch) for p in predictions)/ len(predictions)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model_cy.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += total_loss.item()


    model_cy.eval()
    with torch.no_grad():
        val_preds,_ = model_cy(X_val_t)
        horizons_f1s = []
        for pred in val_preds:
            probs = torch.sigmoid(pred).squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            f1 = f1_score(y_val, preds,zero_division=0)
            horizons_f1s.append(f1)

        avg_f1 = np.mean(horizons_f1s) 

    scheduler.step(avg_f1)


    if(epoch + 1)%10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, loss = {epoch_loss/len(train_loader):.4f}, avg_val_f1 = {avg_f1:.4f}") 

    if avg_f1 > best_val_f1:
        best_val_f1 = avg_f1
        best_epoch = epoch+1 
        no_improve = 0
        torch.save(model_cy.state_dict(), MODELS_PATH + 'model8_cross_year_2013train.pt')

    else: 
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nEarly ")








          














