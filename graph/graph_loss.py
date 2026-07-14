import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and sort the data chronologically
df = pd.read_csv("rapport_federated_learning 12.csv")
df['Horodatage'] = pd.to_datetime(df['Horodatage'])
df = df.sort_values(by='Horodatage')

unique_h = sorted(df['Horodatage'].unique())

# 2. Reconstruct the real-world timeline
start_times = {}
for i, h in enumerate(unique_h):
    if i == 0:
        max_t = df[df['Horodatage'] == h]['Temps_Entrainement_Sec'].max()
        start_times[h] = h - pd.Timedelta(seconds=max_t + 50)
    else:
        start_times[h] = unique_h[i-1]

df['Debut_Round'] = df['Horodatage'].map(start_times)
t0 = start_times[unique_h[0]]

df['Debut_Round_Sec'] = (df['Debut_Round'] - t0).dt.total_seconds()
df['Fin_Entrainement_Sec'] = df['Debut_Round_Sec'] + df['Temps_Entrainement_Sec']
df['Fin_Round_Sec'] = (df['Horodatage'] - t0).dt.total_seconds()

# 3. Create an extended dataset to plot the waiting plateaus properly
extended_rows = []

for h in unique_h:
    round_data = df[df['Horodatage'] == h]
    fin_round_global = (h - t0).total_seconds()
    
    for _, row in round_data.iterrows():
        device = row['Device_ID']
        current_loss = row['Loss_Entrainement']
        t_debut = row['Debut_Round_Sec']
        t_fin_calcul = row['Fin_Entrainement_Sec']
        
        # Check if it's the very first round
        if h == unique_h[0]:
            # For Round 1, start plotting directly from the end of its training to avoid arbitrary start values
            extended_rows.append({'Time_Sec': t_fin_calcul, 'Loss': current_loss, 'Device_ID': device})
            extended_rows.append({'Time_Sec': fin_round_global, 'Loss': current_loss, 'Device_ID': device})
        else:
            # For subsequent rounds, look up the actual final loss from the previous round
            prev_loss = df[(df['Device_ID'] == device) & (df['Horodatage'] < h)]['Loss_Entrainement'].iloc[-1]
            
            # Point A: Round start (takes the previous actual loss value)
            extended_rows.append({'Time_Sec': t_debut, 'Loss': prev_loss, 'Device_ID': device})
            
            # Point B: Training end (loss drops to current value)
            extended_rows.append({'Time_Sec': t_fin_calcul, 'Loss': current_loss, 'Device_ID': device})
            
            # Point C: Global round end (loss plateaus horizontally during synchronization wait)
            extended_rows.append({'Time_Sec': fin_round_global, 'Loss': current_loss, 'Device_ID': device})

df_plot = pd.DataFrame(extended_rows)

# 4. Plot the training curves
fig, ax = plt.subplots(figsize=(12, 6))

sns.lineplot(
    data=df_plot, 
    x='Time_Sec', 
    y='Loss', 
    hue='Device_ID', 
    linewidth=2.5,
    palette='tab10',
    ax=ax
)

# Add vertical synchronization markers for each Federated Learning round
for i, h in enumerate(unique_h):
    fin_sec = (h - t0).total_seconds()
    ax.axvline(x=fin_sec, color='blue', linestyle=':', alpha=0.4)
    ax.text(fin_sec - 15, ax.get_ylim()[1] - 0.002, f"Round {i+1}", 
            rotation=90, va='top', ha='right', color='blue', fontsize=9, alpha=0.6)

# Graph styling and English translations
ax.set_title("Federated Learning Training Loss & Synchronization Idle Times", fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel("Elapsed Real Time (seconds)", fontsize=11)
ax.set_ylabel("Training Loss", fontsize=11)
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()