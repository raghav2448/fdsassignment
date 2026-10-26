import os
import re
import csv
import glob
import math
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib

try:  # Running inside Jupyter notebook?
    get_ipython()
    IN_NOTEBOOK = True
except NameError:
    IN_NOTEBOOK = False
    matplotlib.use("Agg")  # Prevent GUI windows when running as a script

import matplotlib.pyplot as plt

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})

# %% [markdown]
# ## 0. Configuration and Helper Functions

# %%
# ---------- EDIT STUDENT METADATA HERE ----------
STUDENT_NAME = "Aditya Kumar Thakur"
STUDENT_ID = "Your Roll No / ID"
COURSE = "Fundamentals of Data Science"
RETAIL_PATH = None  # e.g., "data/Market_Basket_Optimisation.csv" (None = auto-detect inside data/)
# -----------------------------------------------

SEED = 42
DATA_DIR = "data"
OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
for d in (FIG_DIR, TAB_DIR):
    os.makedirs(d, exist_ok=True)

FIGS = {}  # Stores image paths for report generation
R = {}     # Stores values and tables needed by report generator


def find_retail_file(override=None):
    """Locate retail CSV/Excel file in data/ or root directory."""
    if override and os.path.exists(override):
        return override
    keywords = ["market", "basket", "retail", "transaction", "groceries", "online"]
    for folder in (DATA_DIR, "."):
        for p in sorted(glob.glob(os.path.join(folder, "*"))):
            if p.lower().endswith((".csv", ".xlsx", ".xls")) and any(k in os.path.basename(p).lower() for k in keywords):
                return p
    return None


def show(obj, title=None):
    """Utility to print or display dataframes cleanly."""
    if title:
        print(f"\n=== {title} ===")
    if IN_NOTEBOOK:
        from IPython.display import display
        display(obj)
    else:
        print(obj.to_string() if isinstance(obj, (pd.DataFrame, pd.Series)) else obj)


def save_fig(name):
    path = os.path.join(FIG_DIR, name + ".png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()
    FIGS[name] = path


def save_table(df, name):
    df.to_csv(os.path.join(TAB_DIR, name + ".csv"))


RETAIL_PATH = find_retail_file(os.environ.get("RETAIL_CSV", RETAIL_PATH))
print("Retail dataset path:", RETAIL_PATH)
if RETAIL_PATH is None:
    raise FileNotFoundError("No dataset found. Please place your transaction CSV/Excel file in the 'data/' folder.")

# %% [markdown]
# ## Objectives 1–4: Data Understanding, Cleaning, Descriptive Statistics & Visualisation

# %%
NOISE_ITEMS = {"postage", "dotcom postage", "manual", "carriage", "bank charges", "amazon fee", "discount", "samples", "cruk commission"}


def _clean_item(x):
    x = re.sub(r"\s+", " ", str(x)).strip().lower()
    return None if x in ("", "nan", "none") else x


def load_transactions(path):
    """Parses transaction file into a list of item sets. Auto-detects long/basket format."""
    ext = os.path.splitext(path)[1].lower()
    info = {"file": os.path.basename(path)}
    if ext in (".xlsx", ".xls"):
        raw = pd.read_excel(path)
        fmt = "long"
    else:
        with open(path, encoding="utf-8", errors="ignore") as f:
            first = f.readline().lower()
        fmt = "long" if any(k in first for k in ("invoice", "description", "stockcode", "transaction", "customer")) else "basket"
        if fmt == "long":
            try:
                raw = pd.read_csv(path, encoding="utf-8")
            except UnicodeDecodeError:
                raw = pd.read_csv(path, encoding="latin1")
    info["format"] = fmt

    if fmt == "basket":  # One row = full transaction
        with open(path, encoding="utf-8", errors="ignore", newline="") as f:
            rows = list(csv.reader(f))
        info["raw_rows"] = len(rows)
        tx = [sorted({i for i in (_clean_item(v) for v in row) if i}) for row in rows]
    else:  # Long format (Invoice / Line Item per row)
        cols = {c.lower(): c for c in raw.columns}
        inv = next((cols[c] for c in cols if any(k in c for k in ("invoice", "transaction", "order", "basket", "bill"))), None)
        itm = next((cols[c] for c in cols if any(k in c for k in ("description", "item", "product"))), None)
        if inv is None or itm is None:
            raise ValueError(f"Could not identify invoice/item columns in {list(raw.columns)}")
        info["raw_rows"] = len(raw)
        raw = raw.dropna(subset=[inv, itm])
        raw = raw[~raw[inv].astype(str).str.upper().str.startswith("C")]  # Remove cancellations
        qty = next((cols[c] for c in cols if "quantity" in c), None)
        if qty is not None:
            raw = raw[raw[qty] > 0]
        raw[itm] = raw[itm].map(_clean_item)
        raw = raw.dropna(subset=[itm])
        raw = raw[~raw[itm].isin(NOISE_ITEMS)]
        tx = [sorted(set(v)) for v in raw.groupby(inv)[itm].apply(list)]
    
    tx = [t for t in tx if len(t) > 0]
    info["n_transactions"] = len(tx)
    return tx, info


transactions, retail_info = load_transactions(RETAIL_PATH)
print("File metadata:", retail_info)
print("\nSample transactions:")
for t in transactions[:3]:
    print("  ", t)
R["retail_info"] = retail_info

# %%
basket_sizes = pd.Series([len(t) for t in transactions], name="basket size")
item_counts = Counter(i for t in transactions for i in t)
n_tx = len(transactions)
stats = basket_sizes.describe().round(2).to_frame("value")
show(stats, "Descriptive Statistics: Transaction Size (Items per Basket)")
print(f"Total Transactions: {n_tx:,} | Unique Items: {len(item_counts):,} | Mode Basket Size: {int(basket_sizes.mode()[0])}")

top_items = pd.DataFrame(item_counts.most_common(15), columns=["Product", "Transactions"])
top_items["Support (%)"] = (top_items["Transactions"] / n_tx * 100).round(2)
show(top_items, "Top 15 Most Frequent Products")
save_table(top_items, "B_top_products")

R.update(
    n_tx=n_tx,
    n_items=len(item_counts),
    basket_stats=basket_sizes.describe(),
    top_items=top_items,
    mode_size=int(basket_sizes.mode()[0]),
    single_item_share=float((basket_sizes == 1).mean() * 100)
)

# Plots
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(top_items["Product"][::-1], top_items["Transactions"][::-1], color="#2a7de1")
ax.set_title("Top 15 Most Frequently Purchased Products")
ax.set_xlabel("Number of Transactions")
ax.set_ylabel("Product")
save_fig("B1_top_products")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(basket_sizes, bins=range(1, min(int(basket_sizes.max()) + 2, 40)), color="#2a7de1", edgecolor="white", align="left")
ax.axvline(basket_sizes.mean(), color="red", ls="--", label=f"Mean = {basket_sizes.mean():.2f}")
ax.axvline(basket_sizes.median(), color="green", ls=":", label=f"Median = {basket_sizes.median():.0f}")
ax.set_title("Transaction-Size Distribution")
ax.set_xlabel("Items in Transaction")
ax.set_ylabel("Transaction Count")
ax.legend()
save_fig("B2_transaction_size_distribution")

# %% [markdown]
# ## Objectives 5–6: Numerical Feature Encoding & Principal Component Analysis (PCA)

# %%
TOP_N_ITEMS = 100
MAX_SAMPLES = 3000
N_PCA = 10

keep = [i for i, _ in item_counts.most_common(TOP_N_ITEMS)]
coverage = sum(item_counts[i] for i in keep) / sum(item_counts.values()) * 100
mlb = MultiLabelBinarizer(classes=keep)
Xb = mlb.fit_transform(transactions).astype(float)

nonempty = Xb.sum(axis=1) > 0
Xb, tx_kept = Xb[nonempty], [t for t, k in zip(transactions, nonempty) if k]

rng = np.random.RandomState(SEED)
if len(Xb) > MAX_SAMPLES:
    sel = np.sort(rng.choice(len(Xb), MAX_SAMPLES, replace=False))
    Xs = Xb[sel]
else:
    sel = np.arange(len(Xb))
    Xs = Xb

print(f"Binary Feature Matrix: {Xb.shape} (Top {len(keep)} items cover {coverage:.1f}% of occurrences)")
print(f"Clustering Sample Shape: {Xs.shape}")

# PCA fitting
pca_full = PCA(random_state=SEED).fit(Xs)
cum = np.cumsum(pca_full.explained_variance_ratio_) * 100
pca2 = PCA(n_components=2, random_state=SEED).fit(Xs)
P2 = pca2.transform(Xs)
pcaC = PCA(n_components=N_PCA, random_state=SEED).fit(Xs)
Z = pcaC.transform(Xs)

var2 = pca2.explained_variance_ratio_.sum() * 100
varC = pcaC.explained_variance_ratio_.sum() * 100
print(f"Explained Variance: 2 PCs = {var2:.1f}% | {N_PCA} PCs = {varC:.1f}%")

R.update(
    feat_shape=Xb.shape,
    coverage=coverage,
    sample_shape=Xs.shape,
    var2=var2,
    varC=varC,
    n_pca=N_PCA,
    pc1=pca2.explained_variance_ratio_[0] * 100,
    pc2=pca2.explained_variance_ratio_[1] * 100,
    n_for_80=int(np.searchsorted(cum, 80) + 1)
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(range(1, 31), pca_full.explained_variance_ratio_[:30] * 100, "o-")
axes[0].set_title("PCA Scree Plot")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance (%)")

axes[1].plot(range(1, len(cum) + 1), cum, "o-", ms=3)
axes[1].axhline(80, color="red", ls="--", label="80% Threshold")
axes[1].axvline(N_PCA, color="green", ls=":", label=f"{N_PCA} PCs Used")
axes[1].set_title("Cumulative Explained Variance")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Variance (%)")
axes[1].legend()
save_fig("B3_pca_variance")

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(P2[:, 0], P2[:, 1], s=8, alpha=0.4, color="#2a7de1")
ax.set_title("Transactions Projected onto First Two Principal Components")
ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0] * 100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1] * 100:.1f}%)")
save_fig("B4_pca_scatter")

# %% [markdown]
# ## Objectives 7–8: Clustering Execution & Metric Comparison

# %%
def cluster_metrics(Zm, labels):
    mask = labels != -1
    k = len(set(labels[mask]))
    if k < 2 or mask.sum() <= k:
        return np.nan, np.nan, k, int((~mask).sum())
    return (silhouette_score(Zm[mask], labels[mask]), davies_bouldin_score(Zm[mask], labels[mask]), k, int((~mask).sum()))

# --- 1. K-Means ---
km_rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
    s, d, _, _ = cluster_metrics(Z, km.labels_)
    km_rows.append({"k": k, "inertia": km.inertia_, "silhouette": s, "davies_bouldin": d})
km_tab = pd.DataFrame(km_rows).set_index("k")
show(km_tab.round(4), "K-Means Evaluation for k = 2..8")
K_BEST = int(km_tab["silhouette"].idxmax())
print("Selected k (Highest Silhouette):", K_BEST)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].plot(km_tab.index, km_tab["inertia"], "o-"); axes[0].set_title("K-Means Elbow Plot"); axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
axes[1].plot(km_tab.index, km_tab["silhouette"], "o-", color="green"); axes[1].set_title("Silhouette vs k"); axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette")
axes[2].plot(km_tab.index, km_tab["davies_bouldin"], "o-", color="red"); axes[2].set_title("Davies–Bouldin vs k"); axes[2].set_xlabel("k"); axes[2].set_ylabel("Davies–Bouldin")
save_fig("B5_kmeans_selection")
R.update(km_tab=km_tab, k_best=K_BEST)

labels_all, params = {}, {}
labels_all["K-Means"] = KMeans(n_clusters=K_BEST, n_init=10, random_state=SEED).fit_predict(Z)
params["K-Means"] = f"k={K_BEST}"

# --- 2. Hierarchical (Ward) ---
labels_all["Hierarchical"] = AgglomerativeClustering(n_clusters=K_BEST, linkage="ward").fit_predict(Z)
params["Hierarchical"] = f"Ward linkage, k={K_BEST}"

sub = Z[np.random.RandomState(SEED).choice(len(Z), min(200, len(Z)), replace=False)]
fig, ax = plt.subplots(figsize=(11, 5))
dendrogram(linkage(sub, "ward"), truncate_mode="lastp", p=25, ax=ax, leaf_rotation=90)
ax.set_title("Hierarchical Dendrogram (Ward Linkage, 200 Samples)")
ax.set_xlabel("Cluster Index / Size")
ax.set_ylabel("Ward Distance")
save_fig("B6_dendrogram")

# --- 3. DBSCAN Grid Search ---
best_db, db_rows = None, []
for ms in (5, 10, 20):
    dist = NearestNeighbors(n_neighbors=ms).fit(Z).kneighbors(Z)[0][:, -1]
    if ms == 10:
        kdist = np.sort(dist)
    for q in (40, 50, 60, 70, 80, 90, 95):
        eps = float(np.percentile(dist, q))
        lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(Z)
        s, d, k, noise = cluster_metrics(Z, lab)
        db_rows.append({"eps": round(eps, 4), "min_samples": ms, "clusters": k, "noise": noise, "silhouette": s, "davies_bouldin": d})
        if k >= 2 and noise / len(Z) <= 0.5 and not np.isnan(s) and (best_db is None or s > best_db[0]):
            best_db = (s, eps, ms, lab)

db_tab = pd.DataFrame(db_rows)
show(db_tab.round(4), "DBSCAN Grid Search Summary")
if best_db is not None:
    labels_all["DBSCAN"] = best_db[3]
    params["DBSCAN"] = f"eps={best_db[1]:.3f}, min_samples={best_db[2]}"
else:
    print("DBSCAN failed to discover distinct clusters with <=50% noise.")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(kdist); ax.set_title("DBSCAN K-Distance Plot (min_samples = 10)")
ax.set_xlabel("Sorted Data Points"); ax.set_ylabel("Distance to 10th Nearest Neighbor")
save_fig("B7_dbscan_kdistance")

# --- 4. Spectral Clustering ---
try:
    labels_all["Spectral"] = SpectralClustering(
        n_clusters=K_BEST, affinity="nearest_neighbors", n_neighbors=15, assign_labels="kmeans", random_state=SEED
    ).fit_predict(Z)
    params["Spectral"] = f"k={K_BEST}, 15-NN Graph"
except Exception as e:
    print("Spectral Clustering execution failed:", e)

# Summary Evaluation Table
comp_rows = []
for name in ["K-Means", "Hierarchical", "DBSCAN", "Spectral"]:
    if name in labels_all:
        s, d, k, noise = cluster_metrics(Z, labels_all[name])
        comp_rows.append({
            "Algorithm": name, "Parameters": params[name], "Clusters": k, "Noise points": noise,
            "Silhouette (higher=better)": s, "Davies-Bouldin (lower=better)": d
        })

comp = pd.DataFrame(comp_rows).set_index("Algorithm")
valid = comp.dropna(subset=["Silhouette (higher=better)"])
comp["Rank (silhouette)"] = comp["Silhouette (higher=better)"].rank(ascending=False)
comp["Rank (DB)"] = comp["Davies-Bouldin (lower=better)"].rank(ascending=True)
comp["Average rank"] = comp[["Rank (silhouette)", "Rank (DB)"]].mean(axis=1)

show(comp.round(4), "Overall Clustering Algorithm Comparison")
save_table(comp.round(4), "B_clustering_comparison")

best_sil = valid["Silhouette (higher=better)"].idxmax()
best_dbi = valid["Davies-Bouldin (lower=better)"].idxmin()
best_overall = comp["Average rank"].idxmin()

R.update(comp=comp, best_sil=best_sil, best_dbi=best_dbi, best_overall=best_overall)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
comp["Silhouette (higher=better)"].plot.bar(ax=axes[0], color="#2a7de1", rot=0); axes[0].set_title("Silhouette Score (Higher is Better)"); axes[0].set_ylabel("Score")
comp["Davies-Bouldin (lower=better)"].plot.bar(ax=axes[1], color="#e1692a", rot=0); axes[1].set_title("Davies–Bouldin Index (Lower is Better)"); axes[1].set_ylabel("Score")
save_fig("B8_clustering_metrics")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for ax, name in zip(axes.ravel(), ["K-Means", "Hierarchical", "DBSCAN", "Spectral"]):
    if name in labels_all:
        lab = labels_all[name]
        ax.scatter(P2[lab == -1, 0], P2[lab == -1, 1], s=8, c="lightgrey", label="Noise")
        ax.scatter(P2[lab != -1, 0], P2[lab != -1, 1], s=8, c=lab[lab != -1], cmap="tab10", alpha=0.7)
        ax.set_title(f"{name} ({params[name]})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
fig.suptitle("Clusters Projected onto First 2 Principal Components", y=1.0)
save_fig("B9_cluster_scatter")

# --- Cluster Profiling ---
lab = labels_all[best_overall]
overall_share = Xs.mean(axis=0)
prof = []
for c in sorted(set(lab)):
    m = lab == c
    share = Xs[m].mean(axis=0)
    ok = np.where(share >= 0.05)[0]
    if len(ok) == 0:
        ok = np.argsort(-share)[:5]
    ranked = ok[np.argsort(-(share[ok] / (overall_share[ok] + 1e-5)))] [:5]
    prof.append({
        "Cluster": "Noise" if c == -1 else int(c),
        "Transactions": int(m.sum()),
        "Share (%)": round(m.mean() * 100, 1),
        "Avg Basket Size": round(Xs[m].sum(axis=1).mean(), 2),
        "Distinguishing Products (Cluster Share)": "; ".join(f"{keep[j]} ({share[j] * 100:.0f}%)" for j in ranked)
    })
profile = pd.DataFrame(prof).set_index("Cluster")
show(profile, f"Cluster Profiles – Best Model ({best_overall})")
save_table(profile, "B_cluster_profile")
R["profile"] = profile

# %% [markdown]
# ## Automated Report Generation (.docx)

# %%
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "–" if np.isnan(v) else f"{v:.3f}"
    return str(v)


def add_table(doc, df, index=True, font=8.5):
    d = df.reset_index() if index else df.copy()
    t = doc.add_table(rows=1, cols=len(d.columns))
    t.style = "Light Grid Accent 1"
    for j, c in enumerate(d.columns):
        cell = t.rows[0].cells[j]
        cell.text = ""
        r = cell.paragraphs[0].add_run(str(c))
        r.bold = True
        r.font.size = Pt(font)
    for _, row in d.iterrows():
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = ""
            cells[j].paragraphs[0].add_run(_fmt(v)).font.size = Pt(font)
    doc.add_paragraph()


def add_fig(doc, key, caption, width=6.0):
    if key in FIGS:
        doc.add_picture(FIGS[key], width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        p = doc.add_paragraph()
        r = p.add_run(caption)
        r.italic = True
        r.font.size = Pt(9)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def para(doc, text, bold=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    return p


def bullets(doc, items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def build_report(path=os.path.join(OUT_DIR, "Retail_Clustering_Report.docx")):
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(11)
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(1)
        s.top_margin = s.bottom_margin = Inches(0.9)

    doc.add_heading(f"{COURSE} – Online Retail Clustering Report", 0)
    para(doc, f"Student Name: {STUDENT_NAME}    Student ID: {STUDENT_ID}")

    info = R["retail_info"]
    comp = R["comp"]

    doc.add_heading("1. Problem Statement & Objectives", 1)
    para(doc, "The business problem revolves around an online retail store containing transaction logs of items purchased together. "
              "Lacking automated insight into customer basket compositions, the objective is to preprocess transaction logs, apply PCA "
              "for dimensionality reduction, compare four clustering algorithms (K-Means, Hierarchical, DBSCAN, Spectral), and evaluate "
              "results using the Silhouette Coefficient and Davies–Bouldin index.")

    doc.add_heading("2. Data Preprocessing & Descriptive Statistics", 1)
    para(doc, f"The file '{info['file']}' ({info['format']} format) contained {info['raw_rows']:,} raw records. Post-cleaning (removing duplicates, "
              f"cancelled invoices, and non-product line noise), we obtained {R['n_tx']:,} distinct transactions across {R['n_items']:,} unique products.")
    
    bs = R["basket_stats"]
    para(doc, f"Transaction Basket Statistics: Mean size = {bs['mean']:.2f} items, Median = {bs['50%']:.0f}, Std Dev = {bs['std']:.2f}, "
              f"Max Size = {bs['max']:.0f}. Most frequent basket size: {R['mode_size']} item(s). Single-item transactions represent {R['single_item_share']:.1f}% of all baskets.")
    
    add_table(doc, R["top_items"], index=False)
    add_fig(doc, "B1_top_products", "Figure 1: Top 15 Most Purchased Products", 5.2)
    add_fig(doc, "B2_transaction_size_distribution", "Figure 2: Distribution of Basket Sizes", 4.8)

    doc.add_heading("3. Feature Representation & PCA Dimensionality Reduction", 1)
    para(doc, f"Transactions were converted into binary vectors across the top {R['feat_shape'][1]} products (covering {R['coverage']:.1f}% of total items). "
              f"A random sample of {R['sample_shape'][0]:,} rows was extracted for pairwise clustering computation. PCA reveals high sparsity: "
              f"The first 2 components explain {R['var2']:.1f}% of variance, while {R['n_pca']} components capture {R['varC']:.1f}%.")
    add_fig(doc, "B3_pca_variance", "Figure 3: PCA Scree Plot and Cumulative Variance", 6.0)
    add_fig(doc, "B4_pca_scatter", "Figure 4: 2D Projection of Baskets via PCA", 4.3)

    doc.add_heading("4. Clustering Algorithm Comparison", 1)
    para(doc, f"K-Means was evaluated for k=2..8, identifying k={R['k_best']} as optimal via Silhouette Score. Ward Hierarchical and Spectral clustering "
              f"were also executed with k={R['k_best']}. DBSCAN parameter search tuned eps and min_samples.")
    
    add_table(doc, comp[["Parameters", "Clusters", "Noise points", "Silhouette (higher=better)", "Davies-Bouldin (lower=better)"]].round(4))
    add_fig(doc, "B8_clustering_metrics", "Figure 5: Silhouette and Davies-Bouldin Score Comparison", 5.8)
    add_fig(doc, "B9_cluster_scatter", "Figure 6: Visual Comparison of Cluster Allocations on 2D PCA", 6.0)

    doc.add_heading("5. Cluster Profiling & Interpretation", 1)
    para(doc, f"Based on composite evaluation rank, **{R['best_overall']}** achieved the most robust performance. "
              f"The highest silhouette score recorded was {comp.loc[R['best_sil'], 'Silhouette (higher=better)']:.3f} ({R['best_sil']}).")
    add_table(doc, R["profile"], font=8)

    doc.add_heading("6. Limitations & Business Recommendations", 1)
    para(doc, "Limitations", bold=True)
    bullets(doc, [
        "Binary vectors treat item occurrences as presence/absence, ignoring price and quantity details.",
        "Internal metrics (Silhouette/DB) measure geometric distance, which may not always correspond directly to business value on sparse market basket data.",
        "DBSCAN is sensitive to density changes in high-dimensional space."
    ])
    para(doc, "Business Implications", bold=True)
    bullets(doc, [
        "Product Bundling: Leverage distinguishing cluster products to construct cross-selling deals and bundle recommendations.",
        "Inventory Strategy: Ensure high-performing cluster anchors are kept in continuous supply to prevent stockouts."
    ])

    doc.save(path)
    return path


doc_path = build_report()
print(f"\nSUCCESS: Document generated and saved to '{doc_path}'.")