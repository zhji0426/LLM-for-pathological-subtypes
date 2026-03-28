import os
import re
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import to_rgba
from matplotlib import transforms

# warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------
# Style: Nature Communications-friendly
# ----------------------------
MM_TO_INCH = 1 / 25.4
FIG_W_MM = 180
FIG_H_MM = 165  # stay below ~170 mm
FIG_W = FIG_W_MM * MM_TO_INCH
FIG_H = FIG_H_MM * MM_TO_INCH

FONT_FAMILY = "DejaVu Sans"
mpl.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 6.0,
    "axes.titlesize": 7.0,
    "axes.labelsize": 6.5,
    "xtick.labelsize": 5.8,
    "ytick.labelsize": 5.8,
    "legend.fontsize": 5.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

COLORS = {
    "glomerular_lesions": "#4C78A8",
    "tubulointerstitial_lesions": "#F58518",
    "vascular_lesions": "#72B7B2",
    "immunofluorescence": "#B279A2",
    "inflammation_activity_summary": "#9C755F",
    "other": "#7F7F7F",
}

PANEL_A_MAIN_GROUP_COLORS = {
    "Tubulointerstitial": "#D55E00",
    "Glomerular": "#2C6BA0",
    "Vascular": "#2A9D8F",
    "Immunofluorescence": "#9B59B6",
    "Inflammation summary": "#8D6E63",
    "Other": "#6E6E6E",
}

# new finer palette for panel a subgrouping
SUBGROUP_COLORS = {
    "TI chronic tubular": "#E68613",
    "TI chronic interstitial": "#F2A541",
    "TI active inflammatory": "#D95F02",
    "Glom qualitative": "#4C78A8",
    "Glom quantitative": "#7AA6D1",
    "Vasc chronic changes": "#54A6A6",
    "Vasc hyaline": "#86BCB6",
    "IF": "#B279A2",
    "Inflam summary": "#9C755F",
    "Other": "#7F7F7F",
}

COUNTERFACTUAL_COLORS = {
    "delete": "#4C78A8",
    "swap": "#F58518",
    "block": "#54A24B",
}

LABEL_COLORS = {
    0: "#4C78A8",
    1: "#F58518",
}


def clean_feature_name(s: str) -> str:
    if pd.isna(s):
        return s
    s = str(s)
    s = s.replace("tubulointerstitial_lesions_", "TI ")
    s = s.replace("glomerular_lesions_", "Glom ")
    s = s.replace("vascular_lesions_", "Vasc ")
    s = s.replace("immunofluorescence_", "IF ")
    s = s.replace("inflammation_activity_summary_", "Inflam ")
    s = s.replace("_features_", " ")
    s = s.replace("_quantitative_", " ")
    s = s.replace("_qualitative_", " ")
    s = s.replace("_chronic_changes_", " ")
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_section_from_feature(s: str) -> str:
    s = str(s)
    if s.startswith("tubulointerstitial_lesions_"):
        return "tubulointerstitial_lesions"
    if s.startswith("glomerular_lesions_"):
        return "glomerular_lesions"
    if s.startswith("vascular_lesions_"):
        return "vascular_lesions"
    if s.startswith("immunofluorescence_"):
        return "immunofluorescence"
    if s.startswith("inflammation_activity_summary_"):
        return "inflammation_activity_summary"
    return "other"


def get_panel_a_subgroup(raw_feature: str) -> str:
    """
    Second-level category used only for panel a.
    This keeps the biologic structure clearer than section-only grouping.
    """
    s = str(raw_feature)

    if s.startswith("tubulointerstitial_lesions_chronic_tubular_"):
        return "TI chronic tubular"
    if s.startswith("tubulointerstitial_lesions_chronic_interstitial_"):
        return "TI chronic interstitial"
    if s.startswith("tubulointerstitial_lesions_active_inflammatory_"):
        return "TI active inflammatory"

    if s.startswith("glomerular_lesions_qualitative_"):
        return "Glom qualitative"
    if s.startswith("glomerular_lesions_quantitative_"):
        return "Glom quantitative"

    if s.startswith("vascular_lesions_chronic_changes_"):
        return "Vasc chronic changes"
    if s.startswith("vascular_lesions_hyaline_"):
        return "Vasc hyaline"

    if s.startswith("immunofluorescence_"):
        return "IF"
    if s.startswith("inflammation_activity_summary_"):
        return "Inflam summary"
    return "Other"


def get_panel_a_main_group(raw_feature: str) -> str:
    s = str(raw_feature)
    if s.startswith("tubulointerstitial_lesions_"):
        return "Tubulointerstitial"
    if s.startswith("glomerular_lesions_"):
        return "Glomerular"
    if s.startswith("vascular_lesions_"):
        return "Vascular"
    if s.startswith("immunofluorescence_"):
        return "Immunofluorescence"
    if s.startswith("inflammation_activity_summary_"):
        return "Inflammation summary"
    return "Other"


def get_panel_a_group_ranks():
    main_group_order = [
        "Tubulointerstitial",
        "Glomerular",
        "Vascular",
        "Immunofluorescence",
        "Inflammation summary",
        "Other",
    ]
    subgroup_order = [
        "TI chronic tubular",
        "TI chronic interstitial",
        "TI active inflammatory",
        "Glom qualitative",
        "Glom quantitative",
        "Vasc chronic changes",
        "Vasc hyaline",
        "IF",
        "Inflam summary",
        "Other",
    ]
    main_rank = {k: i for i, k in enumerate(main_group_order)}
    subgroup_rank = {k: i for i, k in enumerate(subgroup_order)}
    return main_rank, subgroup_rank


def make_panel_a_feature_label(raw_feature: str, width: int = 24) -> str:
    core = strip_subgroup_prefix(raw_feature)
    return "\n".join(textwrap.wrap(core, width=width))


def strip_subgroup_prefix(raw_feature: str) -> str:
    """Remove the subgroup prefix and keep only the terminal pathologic descriptor."""
    s = str(raw_feature)
    prefixes = [
        "tubulointerstitial_lesions_chronic_tubular_",
        "tubulointerstitial_lesions_chronic_interstitial_",
        "tubulointerstitial_lesions_active_inflammatory_",
        "glomerular_lesions_qualitative_",
        "glomerular_lesions_quantitative_",
        "vascular_lesions_chronic_changes_",
        "vascular_lesions_hyaline_",
        "immunofluorescence_",
        "inflammation_activity_summary_",
        "tubulointerstitial_lesions_",
        "glomerular_lesions_",
        "vascular_lesions_",
    ]
    for p in prefixes:
        if s.startswith(p):
            s = s[len(p):]
            break

    s = s.replace("_features_", " ")
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def make_panel_a_label(raw_feature: str, multiline: bool = True) -> str:
    subgroup = get_panel_a_subgroup(raw_feature)
    core = strip_subgroup_prefix(raw_feature)
    if multiline:
        return f"{subgroup}\n{core}"
    return f"{subgroup} | {core}"


def add_panel_label(ax, label):
    ax.text(-0.14, 1.04, label, transform=ax.transAxes,
            fontsize=7.5, fontweight="bold", va="bottom", ha="left")


def wrap_labels(labels, width=34):
    return ["\n".join(textwrap.wrap(str(x), width=width)) for x in labels]


def summarize_feature_cf(df: pd.DataFrame, feature_col: str, mode_name: str) -> pd.DataFrame:
    out = (
        df.groupby(feature_col)
          .agg(
              n=("support_score", "size"),
              mean_support=("support_score", "mean"),
              mean_abs_support=("support_score", lambda s: np.mean(np.abs(s))),
              neg_rate=("support_score", lambda s: np.mean(s < 0)),
          )
          .reset_index()
          .rename(columns={feature_col: "feature"})
    )
    out["mode"] = mode_name
    return out


def summarize_block_cf(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["section", "block"])
          .agg(
              n=("support_score", "size"),
              mean_support=("support_score", "mean"),
              mean_abs_support=("support_score", lambda s: np.mean(np.abs(s))),
              neg_rate=("support_score", lambda s: np.mean(s < 0)),
          )
          .reset_index()
    )
    out["section_block"] = out["section"] + " | " + out["block"]
    return out.sort_values("mean_abs_support", ascending=False)


def feature_distribution_from_swap(swap: pd.DataFrame, feature: str, top_n=4) -> pd.DataFrame:
    sub = swap.loc[swap["feature"] == feature, ["label", "old_value"]].copy()
    sub["old_value"] = sub["old_value"].fillna("NA").astype(str)
    top_vals = sub["old_value"].value_counts().head(top_n).index.tolist()
    sub["value_grouped"] = np.where(sub["old_value"].isin(top_vals), sub["old_value"], "Other")
    tab = (sub.groupby(["label", "value_grouped"]).size()
             .reset_index(name="n"))
    total = tab.groupby("label")["n"].transform("sum")
    tab["prop"] = tab["n"] / total
    return tab


def encoded_level_table(encoded: pd.DataFrame, raw_feature: str, top_n=6) -> pd.DataFrame:
    sub = encoded.loc[encoded["raw_feature"] == raw_feature].copy()
    if sub.empty:
        return sub

    def parse_level(x):
        x = str(x)
        x = x.split("cat__", 1)[-1]
        prefix = raw_feature + "_"
        if x.startswith(prefix):
            x = x[len(prefix):]
        return x

    sub["level"] = sub["encoded_feature"].map(parse_level)
    sub = sub.sort_values("importance", ascending=False).head(top_n)
    return sub


def export_panel_a_hierarchy(dat: pd.DataFrame, output_prefix: str):
    out_prefix = Path(output_prefix)
    hierarchy = dat[[
        "raw_feature", "panel_a_main_group", "panel_a_subgroup",
        "panel_a_feature_label", "importance"
    ]].copy()
    hierarchy.to_csv(str(out_prefix) + "_panel_a_hierarchy.csv", index=False)


def build_panel_a_caption(dat: pd.DataFrame) -> str:
    main_order = dat["panel_a_main_group"].drop_duplicates().tolist()
    subgroup_order = dat["panel_a_subgroup"].drop_duplicates().tolist()
    top_main = dat.groupby("panel_a_main_group")["importance"].sum().sort_values(ascending=False)
    top_sub = dat.groupby("panel_a_subgroup")["importance"].sum().sort_values(ascending=False)

    main_txt = ", ".join(main_order)
    sub_txt = ", ".join(subgroup_order[:6])
    dominant_main = top_main.index[0] if len(top_main) else "Tubulointerstitial"
    dominant_sub = top_sub.index[0] if len(top_sub) else "TI chronic tubular"

    caption = (
        "a, Raw-feature SHAP ranking displayed in a hierarchical pathology layout. "
        "Bars represent mean absolute SHAP values for the top-ranked raw pathology descriptors. "
        "The coarse color bands and bar outlines indicate the major pathologic domains "
        f"({main_txt}), whereas bar fill colors and within-domain separators indicate second-level subgroups "
        f"({sub_txt}). Y-axis labels show the terminal descriptor after removal of the hierarchical prefix from the original JSON path. "
        f"In this panel, the largest cumulative contribution arose from the {dominant_main.lower()} domain, "
        f"with {dominant_sub.lower()} features forming the most prominent second-level block among the top-ranked descriptors."
    )
    return caption


def export_figure_caption(dat: pd.DataFrame, output_prefix: str):
    out_prefix = Path(output_prefix)
    caption = build_panel_a_caption(dat)
    (out_prefix.parent / (out_prefix.name + "_caption_panel_a.txt")).write_text(caption + "\n", encoding="utf-8")


def build_integrated_figure(
    shap_raw_path,
    shap_encoded_path,
    mapping_path,
    delete_path,
    swap_path,
    block_path,
    output_prefix=r"E:\igan_nephropathy_research2\import_feature_results_v2\integrated_interpretability_figure"
):
    shap_raw = pd.read_csv(shap_raw_path)
    shap_encoded = pd.read_csv(shap_encoded_path)
    mapping = pd.read_csv(mapping_path)
    delete = pd.read_csv(delete_path)
    swap = pd.read_csv(swap_path)
    block = pd.read_csv(block_path)

    shap_raw["section"] = shap_raw["raw_feature"].map(get_section_from_feature)
    shap_raw["clean"] = shap_raw["raw_feature"].map(clean_feature_name)
    shap_raw["panel_a_main_group"] = shap_raw["raw_feature"].map(get_panel_a_main_group)
    shap_raw["panel_a_subgroup"] = shap_raw["raw_feature"].map(get_panel_a_subgroup)
    shap_raw["panel_a_label"] = shap_raw["raw_feature"].map(make_panel_a_label)
    shap_raw["panel_a_feature_label"] = shap_raw["raw_feature"].map(make_panel_a_feature_label)

    delete_sum = summarize_feature_cf(delete, "feature", "delete")
    swap_sum = summarize_feature_cf(swap, "feature", "swap")
    block_sum = summarize_block_cf(block)

    joint = (
        shap_raw.rename(columns={"raw_feature": "feature", "importance": "shap_importance"})
                [["feature", "shap_importance", "section", "panel_a_subgroup"]]
        .merge(delete_sum[["feature", "mean_abs_support", "neg_rate"]]
               .rename(columns={"mean_abs_support": "delete_abs", "neg_rate": "delete_neg_rate"}),
               on="feature", how="left")
        .merge(swap_sum[["feature", "mean_abs_support", "neg_rate"]]
               .rename(columns={"mean_abs_support": "swap_abs", "neg_rate": "swap_neg_rate"}),
               on="feature", how="left")
    )
    joint["delete_abs"] = joint["delete_abs"].fillna(0)
    joint["swap_abs"] = joint["swap_abs"].fillna(0)
    joint["delete_neg_rate"] = joint["delete_neg_rate"].fillna(0)
    joint["swap_neg_rate"] = joint["swap_neg_rate"].fillna(0)
    joint["joint_score"] = (
        (joint["shap_importance"] - joint["shap_importance"].mean()) / (joint["shap_importance"].std(ddof=0) + 1e-12)
        + (joint["delete_abs"] - joint["delete_abs"].mean()) / (joint["delete_abs"].std(ddof=0) + 1e-12)
        + (joint["swap_abs"] - joint["swap_abs"].mean()) / (joint["swap_abs"].std(ddof=0) + 1e-12)
    )
    joint["clean"] = joint["feature"].map(clean_feature_name)

    top_shap = shap_raw.head(12).copy()
    top_blocks = block_sum.head(8).copy()
    top_joint = joint.sort_values("joint_score", ascending=False).head(8).copy()

    # group-aware ordering for panel a
    subgroup_order = [
        "TI chronic tubular",
        "TI chronic interstitial",
        "TI active inflammatory",
        "Glom qualitative",
        "Glom quantitative",
        "Vasc chronic changes",
        "Vasc hyaline",
        "IF",
        "Inflam summary",
        "Other",
    ]
    top_shap["subgroup_rank"] = top_shap["panel_a_subgroup"].map({k: i for i, k in enumerate(subgroup_order)}).fillna(999)
    top_shap = top_shap.sort_values(["subgroup_rank", "importance"], ascending=[True, True]).copy()

    candidate_feats = joint.sort_values("joint_score", ascending=False)["feature"].tolist()
    dist_features = []
    for feat in candidate_feats:
        if (swap["feature"] == feat).any():
            dist_features.append(feat)
        if len(dist_features) == 3:
            break

    fig = plt.figure(figsize=(FIG_W, FIG_H), constrained_layout=False)
    gs = gridspec.GridSpec(
        nrows=3, ncols=4, figure=fig,
        width_ratios=[1.25, 1.05, 1.15, 1.1],
        height_ratios=[1.0, 1.05, 1.0],
        left=0.07, right=0.99, bottom=0.07, top=0.965,
        wspace=0.7, hspace=0.95
    )

    # ----------------------------
    # Panel a: hierarchical SHAP raw top features
    # ----------------------------
    axa = fig.add_subplot(gs[0, 0:2])
    dat = top_shap.copy().reset_index(drop=True)
    y = np.arange(len(dat))
    xmax = max(float(dat["importance"].max()), 1e-6)

    subgroup_fills = [SUBGROUP_COLORS.get(g, SUBGROUP_COLORS["Other"]) for g in dat["panel_a_subgroup"]]
    main_edges = [PANEL_A_MAIN_GROUP_COLORS.get(g, PANEL_A_MAIN_GROUP_COLORS["Other"]) for g in dat["panel_a_main_group"]]

    trans = transforms.blended_transform_factory(axa.transAxes, axa.transData)
    main_sizes = dat.groupby("panel_a_main_group", sort=False).size()
    subgroup_sizes = dat.groupby(["panel_a_main_group", "panel_a_subgroup"], sort=False).size().reset_index(name="n")

    y_cursor = 0
    for main_group, n_main in main_sizes.items():
        y0 = y_cursor - 0.5
        y1 = y_cursor + n_main - 0.5
        axa.axhspan(
            y0, y1,
            facecolor=to_rgba(PANEL_A_MAIN_GROUP_COLORS.get(main_group, PANEL_A_MAIN_GROUP_COLORS["Other"]), 0.08),
            edgecolor="none", zorder=0
        )
        axa.add_patch(Rectangle(
            (-0.165, y0), 0.03, n_main,
            transform=trans,
            facecolor=PANEL_A_MAIN_GROUP_COLORS.get(main_group, PANEL_A_MAIN_GROUP_COLORS["Other"]),
            edgecolor="none", clip_on=False, zorder=3
        ))
        axa.text(
            -0.175, y0 + n_main / 2, main_group,
            transform=trans, ha="right", va="center",
            fontsize=5.8, fontweight="bold", clip_on=False
        )
        if y_cursor > 0:
            axa.axhline(y_cursor - 0.5, color="#B8B8B8", lw=1.0, zorder=1)
        y_cursor += n_main

    axa.barh(y, dat["importance"], color=subgroup_fills,
             edgecolor=main_edges, linewidth=1.3, zorder=2)

    sub_cursor = 0
    for _, row in subgroup_sizes.iterrows():
        n_sub = int(row["n"])
        subgroup = row["panel_a_subgroup"]
        center = sub_cursor + (n_sub - 1) / 2
        axa.text(
            -0.02, center, subgroup,
            transform=trans, ha="right", va="center",
            fontsize=5.1, color=SUBGROUP_COLORS.get(subgroup, SUBGROUP_COLORS["Other"]),
            fontweight="bold", clip_on=False
        )
        if sub_cursor > 0:
            axa.axhline(sub_cursor - 0.5, color="#D7D7D7", lw=0.7, ls="--", zorder=1)
        sub_cursor += n_sub

    axa.set_yticks(y)
    axa.set_yticklabels(dat["panel_a_feature_label"])
    axa.tick_params(axis="y", pad=2)
    axa.set_xlim(0, xmax * 1.12)
    axa.set_xlabel("Mean |SHAP value|")
    axa.set_title("Raw-feature SHAP ranking by major domain and second-level subgroup")
    axa.spines[["top", "right"]].set_visible(False)
    add_panel_label(axa, "a")

    main_present = dat["panel_a_main_group"].drop_duplicates().tolist()
    subgroup_present = dat["panel_a_subgroup"].drop_duplicates().tolist()
    main_handles = [Patch(facecolor=PANEL_A_MAIN_GROUP_COLORS[g], edgecolor=PANEL_A_MAIN_GROUP_COLORS[g]) for g in main_present]
    sub_handles = [Patch(facecolor=SUBGROUP_COLORS[g], edgecolor="none") for g in subgroup_present]

    leg1 = axa.legend(
        main_handles, main_present, title="Major domain", frameon=False,
        ncols=2, loc="upper right", bbox_to_anchor=(1.00, 1.02),
        borderpad=0.2, handlelength=1.0, columnspacing=0.8, title_fontsize=5.8
    )
    axa.add_artist(leg1)
    axa.legend(
        sub_handles, subgroup_present, title="Second-level subgroup", frameon=False,
        ncols=2, loc="lower right", bbox_to_anchor=(1.00, -0.02),
        borderpad=0.2, handlelength=1.0, columnspacing=0.8, title_fontsize=5.8
    )

    export_panel_a_hierarchy(dat, output_prefix)
    export_figure_caption(dat, output_prefix)

    # ----------------------------
    # Panel b: block-level counterfactual
    # ----------------------------
    axb = fig.add_subplot(gs[0, 2])
    bdat = top_blocks.sort_values("mean_abs_support", ascending=True)
    bcolors = [COLORS.get(sec, COLORS["other"]) for sec in bdat["section"]]
    labels = [clean_feature_name(x.replace("|", " | ")) for x in bdat["section_block"]]
    axb.barh(labels, bdat["mean_abs_support"], color=bcolors, edgecolor="none")
    axb.set_xlabel("Mean |support score|")
    axb.set_title("Block-level replacement")
    axb.spines[["top", "right"]].set_visible(False)
    add_panel_label(axb, "b")

    # ----------------------------
    # Panel c: joint evidence bubble plot
    # ----------------------------
    axc = fig.add_subplot(gs[0, 3])
    cdat = top_joint.copy().sort_values("joint_score", ascending=False)
    ccolors = [COLORS.get(sec, COLORS["other"]) for sec in cdat["section"]]
    sizes = 40 + 900 * (cdat["delete_abs"] / (cdat["delete_abs"].max() + 1e-12))
    axc.scatter(cdat["shap_importance"], cdat["swap_abs"], s=sizes, c=ccolors,
                alpha=0.85, linewidth=0.4, edgecolor="black")
    for _, row in cdat.iterrows():
        axc.text(row["shap_importance"] * 1.01, row["swap_abs"] * 1.005,
                 clean_feature_name(row["feature"]), fontsize=5.0)
    axc.set_xlabel("SHAP importance")
    axc.set_ylabel("Swap mean |support|")
    axc.set_title("Joint evidence")
    axc.spines[["top", "right"]].set_visible(False)
    add_panel_label(axc, "c")

    # ----------------------------
    # Panel d/e/f: value distributions for top 3 features
    # ----------------------------
    dist_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    value_palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#9D755D"]
    for ax, feat, panel in zip(dist_axes, dist_features, ["d", "e", "f"]):
        tab = feature_distribution_from_swap(swap, feat, top_n=4)
        pivot = (tab.pivot(index="label", columns="value_grouped", values="prop")
                   .fillna(0))
        cols = pivot.sum(axis=0).sort_values(ascending=False).index.tolist()
        pivot = pivot[cols]
        bottom = np.zeros(len(pivot))
        x = np.arange(len(pivot.index))
        for i, col in enumerate(pivot.columns):
            ax.bar(x, pivot[col].values, bottom=bottom, color=value_palette[i % len(value_palette)],
                   edgecolor="white", linewidth=0.4, width=0.65, label=str(col))
            bottom += pivot[col].values
        ax.set_xticks(x)
        ax.set_xticklabels([f"Class {int(v)}" for v in pivot.index])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Proportion")
        ax.set_title(clean_feature_name(feat), fontsize=6.4)
        ax.spines[["top", "right"]].set_visible(False)
        add_panel_label(ax, panel)
        if panel == "f":
            ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left",
                      title="Value", title_fontsize=5.8)

    # ----------------------------
    # Panel g: encoded-level SHAP for top raw feature
    # ----------------------------
    axg = fig.add_subplot(gs[1, 3])
    top_raw_feature = shap_raw.iloc[0]["raw_feature"]
    edat = encoded_level_table(shap_encoded, top_raw_feature, top_n=6)
    if not edat.empty:
        edat = edat.sort_values("importance", ascending=True)
        axg.barh(edat["level"], edat["importance"], color=COLORS[get_section_from_feature(top_raw_feature)])
        axg.set_xlabel("Mean |SHAP value|")
        axg.set_title(f"Encoded levels:\n{clean_feature_name(top_raw_feature)}")
        axg.spines[["top", "right"]].set_visible(False)
    add_panel_label(axg, "g")

    # ----------------------------
    # Panel h: method concordance heatmap
    # ----------------------------
    axh = fig.add_subplot(gs[2, 0:2])
    hdat = top_joint.copy().sort_values("joint_score", ascending=True)
    mat = hdat[["shap_importance", "delete_abs", "swap_abs"]].to_numpy(dtype=float)
    mat = (mat - mat.min(axis=0, keepdims=True)) / (np.ptp(mat, axis=0, keepdims=True) + 1e-12)
    im = axh.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axh.set_xticks([0, 1, 2])
    axh.set_xticklabels(["SHAP", "Delete", "Swap"])
    axh.set_yticks(np.arange(len(hdat)))
    axh.set_yticklabels(wrap_labels(hdat["clean"], width=32))
    axh.set_title("Concordance across explanation methods")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            axh.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                     fontsize=4.9, color=("white" if mat[i, j] > 0.55 else "black"))
    cbar = fig.colorbar(im, ax=axh, fraction=0.03, pad=0.02)
    cbar.set_label("Column-normalized score")
    add_panel_label(axh, "h")

    # ----------------------------
    # Panel i: mapping summary
    # ----------------------------
    axi = fig.add_subplot(gs[2, 2])
    mdat = mapping["status"].fillna("unmatched").value_counts().rename_axis("status").reset_index(name="n")
    total = mdat["n"].sum()
    mdat["prop"] = mdat["n"] / total
    color_map = {"matched": "#4C78A8", "unmatched": "#BDBDBD"}
    axi.bar(mdat["status"], mdat["prop"], color=[color_map.get(s, "#999999") for s in mdat["status"]],
            width=0.6, edgecolor="none")
    for i, row in mdat.iterrows():
        axi.text(i, row["prop"] + 0.02, f"{row['n']}/{total}", ha="center", va="bottom", fontsize=5.6)
    axi.set_ylim(0, 1.08)
    axi.set_ylabel("Proportion")
    axi.set_title("SHAP-to-JSON mapping")
    axi.spines[["top", "right"]].set_visible(False)
    add_panel_label(axi, "i")

    # ----------------------------
    # Panel j: interpretation summary box
    # ----------------------------
    axj = fig.add_subplot(gs[2, 3])
    axj.axis("off")
    text = (
        "Integrated interpretation:\n"
        "• SHAP and counterfactual analyses converge on tubulointerstitial descriptors.\n"
        "• Chronic tubulointerstitial injury blocks show the largest replacement effects.\n"
        "• Tubular atrophy, interstitial fibrosis, and inflammatory infiltration are the most reproducible drivers.\n"
        "• Immunofluorescence contributes minimally in this latent 2-class separation."
    )
    axj.text(0.02, 0.98, text, ha="left", va="top", fontsize=6.0,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7F7F7", edgecolor="#D9D9D9"))
    add_panel_label(axj, "j")

    out_prefix = Path(output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_prefix) + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(str(out_prefix) + ".svg", bbox_inches="tight")
    fig.savefig(str(out_prefix) + ".png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    joint.sort_values("joint_score", ascending=False).to_csv(str(out_prefix) + "_joint_feature_summary.csv", index=False)
    block_sum.to_csv(str(out_prefix) + "_block_summary.csv", index=False)

    # helpful export for manuscript/result writing
    top_shap[["raw_feature", "panel_a_subgroup", "panel_a_label", "importance"]].to_csv(
        str(out_prefix) + "_panel_a_second_level_labels.csv", index=False
    )

def get_base_dir() -> str:
    """获取当前脚本所在目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    data_dir = get_base_dir()
    data_dir = Path(os.path.join(data_dir, r"import_feature_results\tables"))
    build_integrated_figure(
        shap_raw_path=data_dir / "shap_top_features_raw.csv",
        shap_encoded_path=data_dir / "shap_top_features_encoded.csv",
        mapping_path=data_dir / "shap_to_json_feature_mapping.csv",
        delete_path=data_dir / "counterfactual_delete.csv",
        swap_path=data_dir / "counterfactual_swap.csv",
        block_path=data_dir / "counterfactual_block.csv",
        output_prefix=data_dir / "integrated_interpretability_figure_v3",
    )


if __name__ == "__main__":
    main()
