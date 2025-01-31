import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast  # Per convertire le stringhe di liste in liste vere

# Caricare i dati
file_path = "resultsAdam.xlsx"  # Modifica il percorso se necessario
df = pd.read_excel(file_path)

# Convertire colonne con liste in valori numerici (media)
if "Num Neighbors" in df.columns:
    df["Num Neighbors"] = df["Num Neighbors"].apply(lambda x: sum(ast.literal_eval(x)) / len(ast.literal_eval(x)) if isinstance(x, str) else x)

# Impostare lo stile dei grafici
sns.set(style="whitegrid")

# Creare istogrammi per le metriche
metrics = ["AUC", "F1-score", "Precision", "Recall", "Loss"]
plt.figure(figsize=(15, 8))
for i, metric in enumerate(metrics):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df[metric], bins=20, kde=True)
    plt.title(f"Distribuzione di {metric}")
plt.tight_layout()
plt.show()

# Creare box plot per le metriche
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[metrics])
plt.title("Box Plot delle metriche")
plt.xticks(rotation=45)
plt.show()

# Scatter plot tra iperparametri e metriche
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

sns.scatterplot(x=df["Hidden Channels"], y=df["AUC"], hue=df["Neg Sampling Ratio"], palette="viridis", ax=axes[0, 0])
axes[0, 0].set_title("Hidden Channels vs AUC")

sns.scatterplot(x=df["Learning Rate"], y=df["Loss"], hue=df["Batch Size"], palette="coolwarm", ax=axes[0, 1])
axes[0, 1].set_title("Learning Rate vs Loss")

sns.scatterplot(x=df["Batch Size"], y=df["F1-score"], hue=df["Neg Sampling Ratio"], palette="magma", ax=axes[1, 0])
axes[1, 0].set_title("Batch Size vs F1-score")

sns.scatterplot(x=df["Neg Sampling Ratio"], y=df["Precision"], hue=df["Hidden Channels"], palette="cividis", ax=axes[1, 1])
axes[1, 1].set_title("Neg Sampling Ratio vs Precision")

plt.tight_layout()
plt.show()

# Heatmap delle correlazioni tra le variabili (escludendo colonne non numeriche)
plt.figure(figsize=(10, 8))
corr_matrix = df.select_dtypes(include=["number"]).corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Heatmap delle Correlazioni")
plt.show()
