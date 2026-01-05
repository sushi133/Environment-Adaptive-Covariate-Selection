"""
Generate LaTeX tabular snippets for the selector-accuracy tables in the paper.

Inputs (CSV):
  - sim_a_env_selector_acc.csv : varies training environments and sigma
  - sim_a_sam_selector_acc.csv : varies samples per env and sigma 
  - sim_a_lev_selector_acc.csv : varies max perturbation level 
  - sim_a_sta_selector_acc.csv : varies summary representation 

Outputs (LaTeX):
  - env_and_sam_u_tabular.tex  
  - env_and_sam_u+c_tabular.tex     
  - range_vs_repr_u_tabular.tex       
  - range_vs_repr_u+c_tabular.tex   

python make_latex_tables_selector_acc.py \
  --env_csv sim_a_env_selector_acc.csv \
  --sam_csv sim_a_sam_selector_acc.csv \
  --lev_csv sim_a_lev_selector_acc.csv \
  --sta_csv sim_a_sta_selector_acc.csv \
  --out_dir auto_tables   
  
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def fmt3(x) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.3f}"


def fmt1(x) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.1f}"


def require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}. Found: {df.columns.tolist()}")


def add_latex_labels(sta_df: pd.DataFrame) -> pd.DataFrame:
    repr_map = {
        "all": r"${r_e,s_{2,e},s_{3,e}}$",
        "corr(c2,x)": r"$r_e$",
        "std(c2)": r"$s_{2,e}$",
        "std(x)": r"$s_{3,e}$",
    }
    sta_df = sta_df.copy()
    sta_df["latex_label"] = sta_df["representation"].map(repr_map).fillna(sta_df["representation"])
    return sta_df


def render_pick_env_and_sam_twocol(env_df: pd.DataFrame, sam_df: pd.DataFrame) -> str:
    # u-only table: (sigma=1,5,10) for env + sam
    env_df = env_df.sort_values("n_env", ascending=False).reset_index(drop=True)
    sam_df = sam_df.sort_values("samples_per_env", ascending=False).reset_index(drop=True)

    n = max(len(env_df), len(sam_df))
    rows: list[str] = []
    for i in range(n):
        left = env_df.iloc[i] if i < len(env_df) else None
        right = sam_df.iloc[i] if i < len(sam_df) else None

        if left is not None:
            lvals = [int(left["n_env"]), fmt3(left["sigma1_u"]), fmt3(left["sigma5_u"]), fmt3(left["sigma10_u"])]
        else:
            lvals = ["", "", "", ""]
        if right is not None:
            rvals = [
                int(right["samples_per_env"]),
                fmt3(right["sigma1_u"]),
                fmt3(right["sigma5_u"]),
                fmt3(right["sigma10_u"]),
            ]
        else:
            rvals = ["", "", "", ""]

        rows.append(f"{lvals[0]} & {lvals[1]} & {lvals[2]} & {lvals[3]}   & {rvals[0]} & {rvals[1]} & {rvals[2]} & {rvals[3]} \\\\")

    body = "\n".join(rows)
    return r"""\begin{tabular}{
  @{} r
  @{\hspace{1.4em}} S[table-format=1.3]
  @{\hspace{1.4em}} S[table-format=1.3]
  @{\hspace{1.4em}} S[table-format=1.3]
  @{\hspace{1.4em}} r
  @{\hspace{1.4em}} S[table-format=1.3]
  @{\hspace{1.4em}} S[table-format=1.3]
  @{\hspace{1.4em}} S[table-format=1.3] @{}
}
\toprule
\multicolumn{4}{c}{Training environments} &
\multicolumn{4}{c}{Samples per environment} \\
\cmidrule(lr){1-4} \cmidrule(lr){5-8}
& {$\sigma=1$} & {$\sigma=5$} & {$\sigma=10$}
& & {$\sigma=1$} & {$\sigma=5$} & {$\sigma=10$} \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
"""


def render_pick_sigma_side_by_side(env_df: pd.DataFrame, sam_df: pd.DataFrame) -> str:
    # u & c table: (sigma=1,5,10) for env + sam
    env_df = env_df.sort_values("n_env", ascending=False).reset_index(drop=True)
    sam_df = sam_df.sort_values("samples_per_env", ascending=False).reset_index(drop=True)

    n = max(len(env_df), len(sam_df))
    rows: list[str] = []
    for i in range(n):
        left = env_df.iloc[i] if i < len(env_df) else None
        right = sam_df.iloc[i] if i < len(sam_df) else None

        if left is not None:
            lvals = [
                int(left["n_env"]),
                fmt3(left["sigma1_u"]), fmt3(left["sigma1_c"]),
                fmt3(left["sigma5_u"]), fmt3(left["sigma5_c"]),
                fmt3(left["sigma10_u"]), fmt3(left["sigma10_c"]),
            ]
        else:
            lvals = [""] * 7

        if right is not None:
            rvals = [
                int(right["samples_per_env"]),
                fmt3(right["sigma1_u"]), fmt3(right["sigma1_c"]),
                fmt3(right["sigma5_u"]), fmt3(right["sigma5_c"]),
                fmt3(right["sigma10_u"]), fmt3(right["sigma10_c"]),
            ]
        else:
            rvals = [""] * 7

        rows.append(
            f"{lvals[0]} & {lvals[1]} & {lvals[2]} & {lvals[3]} & {lvals[4]} & {lvals[5]} & {lvals[6]}   "
            f"& {rvals[0]} & {rvals[1]} & {rvals[2]} & {rvals[3]} & {rvals[4]} & {rvals[5]} & {rvals[6]} \\\\"
        )

    body = "\n".join(rows)
    return r"""\begin{tabular}{
  @{} r                             % x (left block)
  @{\hspace{1.2em}} S[table-format=1.3] @{\hspace{1.2em}} S[table-format=1.3]  % σ=1 (U,C)
  @{\hspace{1.2em}} S[table-format=1.3] @{\hspace{1.2em}} S[table-format=1.3]  % σ=5 (U,C)
  @{\hspace{1.2em}} S[table-format=1.3] @{\hspace{1.2em}} S[table-format=1.3]  % σ=10 (U,C)
  @{\hspace{1.8em}}                 % gap between left/right blocks
  r                                 % x (right block)
  @{\hspace{1.2em}} S[table-format=1.3] @{\hspace{1.2em}} S[table-format=1.3]  % σ=1 (U,C)
  @{\hspace{1.2em}} S[table-format=1.3] @{\hspace{1.2em}} S[table-format=1.3]  % σ=5 (U,C)
  @{\hspace{1.2em}} S[table-format=1.3] @{\hspace{1.2em}} S[table-format=1.3]  % σ=10 (U,C)
  @{}}
\toprule
% Block titles: each block spans 7 columns (x + 6 data columns)
\multicolumn{7}{c}{Training environments} &
\multicolumn{7}{c}{Samples per environment} \\
\cmidrule(lr){1-7}\cmidrule(lr){8-14}
% Sigma headers: each over exactly (U,C)
 & \multicolumn{2}{c}{$\sigma=1$} & \multicolumn{2}{c}{$\sigma=5$} & \multicolumn{2}{c}{$\sigma=10$}
 &  & \multicolumn{2}{c}{$\sigma=1$} & \multicolumn{2}{c}{$\sigma=5$} & \multicolumn{2}{c}{$\sigma=10$} \\
% U/C row: sits directly in the S columns that hold the numbers
 & {u} & {c} & {u} & {c} & {u} & {c}
 &   & {u} & {c} & {u} & {c} & {u} & {c} \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
"""


def render_supp_pick_range_vs_repr(lev_df: pd.DataFrame, sta_df: pd.DataFrame) -> str:
    # u-only (supp) table: max_level on left; representation on right
    lev_df = lev_df.sort_values("max_level", ascending=False).reset_index(drop=True)

    desired_order = ["all", "corr(c2,x)", "std(c2)", "std(x)"]
    sta_df = sta_df.copy()
    sta_df["order"] = sta_df["representation"].apply(lambda x: desired_order.index(x) if x in desired_order else 999)
    sta_df = sta_df.sort_values("order").reset_index(drop=True)

    n = max(len(lev_df), len(sta_df))
    rows: list[str] = []
    for i in range(n):
        left = lev_df.iloc[i] if i < len(lev_df) else None
        right = sta_df.iloc[i] if i < len(sta_df) else None

        if left is not None:
            l_level = fmt1(left["max_level"])
            l_u = fmt3(left["u"])
        else:
            l_level, l_u = "", ""

        if right is not None:
            r_label = right["latex_label"]
            r_u = fmt3(right["u"])
        else:
            r_label, r_u = "", ""

        rows.append(f"{l_level} & {l_u}   & {r_label} & {r_u} \\\\")

    body = "\n".join(rows)
    return r"""\begin{tabular}{
  @{} r
  @{\hspace{1.4em}} S[table-format=1.3]
  @{\hspace{2.0em}}
  l
  @{\hspace{1.4em}} S[table-format=1.3] @{}
}
\toprule
\multicolumn{2}{c}{Training coverage} &
\multicolumn{2}{c}{Summaries} \\
\cmidrule(lr){1-2}\cmidrule(lr){3-4}
""" + body + r"""
\bottomrule
\end{tabular}
"""


def render_supp_pick_causal_vs_unc(lev_df: pd.DataFrame, sta_df: pd.DataFrame) -> str:
    # u & c (supp) table: max_level on left; representation on right
    lev_df = lev_df.sort_values("max_level", ascending=False).reset_index(drop=True)

    desired_order = ["all", "corr(c2,x)", "std(c2)", "std(x)"]
    sta_df = sta_df.copy()
    sta_df["order"] = sta_df["representation"].apply(lambda x: desired_order.index(x) if x in desired_order else 999)
    sta_df = sta_df.sort_values("order").reset_index(drop=True)

    n = max(len(lev_df), len(sta_df))
    rows: list[str] = []
    for i in range(n):
        left = lev_df.iloc[i] if i < len(lev_df) else None
        right = sta_df.iloc[i] if i < len(sta_df) else None

        if left is not None:
            l_level = fmt1(left["max_level"])
            l_u = fmt3(left["u"])
            l_c = fmt3(left["c"])
        else:
            l_level, l_u, l_c = "", "", ""

        if right is not None:
            r_label = right["latex_label"]
            r_u = fmt3(right["u"])
            r_c = fmt3(right["c"])
        else:
            r_label, r_u, r_c = "", "", ""

        rows.append(f"{l_level} & {l_u} & {l_c}   & {r_label} & {r_u} & {r_c} \\\\")

    body = "\n".join(rows)
    return r"""\begin{tabular}{
  @{} r                             % left block: level
  @{\hspace{1.2em}} S[table-format=1.3] @{\hspace{1.2em}} S[table-format=1.3]  % (U, C)
  @{\hspace{1.8em}}                 % gap between blocks
  l                                 % right block: label
  @{\hspace{1.2em}} S[table-format=1.3] @{\hspace{1.2em}} S[table-format=1.3]  % (U, C)
  @{}}
\toprule
\multicolumn{3}{c}{Training coverage} &
\multicolumn{3}{c}{Summaries} \\
\cmidrule(lr){1-3}\cmidrule(lr){4-6}
 & {u} & {c} &  & {u} & {c} \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_csv", required=True, type=Path)
    ap.add_argument("--sam_csv", required=True, type=Path)
    ap.add_argument("--lev_csv", required=True, type=Path)
    ap.add_argument("--sta_csv", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()

    env_df = pd.read_csv(args.env_csv)
    sam_df = pd.read_csv(args.sam_csv)
    lev_df = pd.read_csv(args.lev_csv)
    sta_df = pd.read_csv(args.sta_csv)

    require_cols(env_df, ["n_env", "sigma1_u", "sigma1_c", "sigma5_u", "sigma5_c", "sigma10_u", "sigma10_c"], "env_csv")
    require_cols(sam_df, ["samples_per_env", "sigma1_u", "sigma1_c", "sigma5_u", "sigma5_c", "sigma10_u", "sigma10_c"], "sam_csv")
    require_cols(lev_df, ["max_level", "u", "c"], "lev_csv")
    require_cols(sta_df, ["representation", "u", "c"], "sta_csv")

    sta_df = add_latex_labels(sta_df)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    (out / "env_and_sam_u_tabular.tex").write_text(render_pick_env_and_sam_twocol(env_df, sam_df))
    (out / "env_and_sam_u+c_tabular.tex").write_text(render_pick_sigma_side_by_side(env_df, sam_df))
    (out / "range_vs_repr_u_tabular.tex").write_text(render_supp_pick_range_vs_repr(lev_df, sta_df))
    (out / "range_vs_repr_u+c_tabular.tex").write_text(render_supp_pick_causal_vs_unc(lev_df, sta_df))

    print(f"Wrote 4 LaTeX tabular snippets to: {out.resolve()}")


if __name__ == "__main__":
    main()
