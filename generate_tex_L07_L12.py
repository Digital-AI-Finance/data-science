"""Generate .tex files for L07-L12"""
from pathlib import Path

BASE_DIR = Path(__file__).parent

PREAMBLE = r"""\documentclass[8pt,aspectratio=169]{beamer}
\usetheme{Madrid}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}

\definecolor{mlpurple}{RGB}{51,51,178}
\definecolor{mllavender}{RGB}{173,173,224}
\definecolor{mllavender2}{RGB}{193,193,232}
\definecolor{mllavender3}{RGB}{204,204,235}
\definecolor{mllavender4}{RGB}{214,214,239}
\definecolor{mlorange}{RGB}{255, 127, 14}
\definecolor{mlgreen}{RGB}{44, 160, 44}

\setbeamercolor{palette primary}{bg=mllavender3,fg=mlpurple}
\setbeamercolor{structure}{fg=mlpurple}
\setbeamercolor{frametitle}{fg=mlpurple,bg=mllavender3}

\setbeamertemplate{navigation symbols}{}
\setbeamersize{text margin left=5mm,text margin right=5mm}

\newcommand{\bottomnote}[1]{\vfill\footnotesize\textbf{#1}}

"""

LESSONS = {
    "L07_Missing_Data": (
        "Lesson 07: Missing Data and Cleaning",
        ["isna()/isnull() for detection", "fillna() methods (ffill, bfill, mean)",
         "dropna() to remove missing", "Handling duplicates", "Data type conversion"],
        ["01_missing_patterns", "02_fillna_methods", "03_quality_checklist", "04_imputation",
         "05_dropna", "06_duplicates", "07_cleaning_workflow", "08_before_after"]
    ),
    "L08_Basic_Operations": (
        "Lesson 08: Basic Operations",
        ["Creating new columns", "apply() for transformations", "Arithmetic operations",
         "Sorting with sort_values()", "Calculating returns and moving averages"],
        ["01_column_creation", "02_apply_function", "03_arithmetic", "04_sorting",
         "05_value_counts", "06_returns", "07_moving_average", "08_cheat_sheet"]
    ),
    "L09_GroupBy_Operations": (
        "Lesson 09: GroupBy Operations",
        ["Split-apply-combine paradigm", "groupby() basics", "Aggregation functions",
         "transform() vs agg()", "Multi-column grouping"],
        ["01_split_apply_combine", "02_groupby_workflow", "03_aggregation_functions",
         "04_transform_vs_agg", "05_multi_groupby", "06_sector_analysis",
         "07_groupby_patterns", "08_finance_groupby"]
    ),
    "L10_Merging_Joining": (
        "Lesson 10: Merging and Joining",
        ["pd.concat() for stacking", "pd.merge() for SQL-style joins",
         "Join types: inner, outer, left, right", "Handling key columns"],
        ["01_concat", "02_merge_types", "03_join_comparison", "04_merge_workflow",
         "05_key_matching", "06_finance_merge", "07_multi_source", "08_troubleshooting"]
    ),
    "L11_NumPy_Basics": (
        "Lesson 11: NumPy Basics",
        ["Arrays vs lists", "Vectorized operations", "Broadcasting",
         "Mathematical functions", "Portfolio calculations"],
        ["01_array_vs_list", "02_vectorization", "03_broadcasting", "04_array_ops",
         "05_math_functions", "06_portfolio_weights", "07_correlation", "08_numpy_finance"]
    ),
    "L12_Time_Series": (
        "Lesson 12: Time Series Basics",
        ["DateTime index", "Resampling (daily to monthly)", "Rolling windows",
         "shift() and pct_change()", "Time series patterns in finance"],
        ["01_time_series", "02_datetime_parsing", "03_resampling", "04_rolling_window",
         "05_shift_lag", "06_pct_change", "07_components", "08_patterns"]
    )
}

def generate_lesson_tex(folder, title, objectives, charts):
    """Generate a lesson .tex file"""

    # Build slides content
    slides = []

    # Title slide
    slides.append(r"""
\title{%s}
\subtitle{Data Science with Python -- BSc Course}
\date{45 Minutes}

\begin{document}

\begin{frame}[plain]
\titlepage
\end{frame}
""" % title)

    # Learning objectives
    obj_items = "\n".join([f"\\item {obj}" for obj in objectives])
    slides.append(r"""
\begin{frame}[t]{Learning Objectives}
\textbf{After this lesson, you will be able to:}
\begin{itemize}
%s
\end{itemize}
\bottomnote{Finance application: Stock data processing and analysis}
\end{frame}
""" % obj_items)

    # Chart slides
    for i, chart in enumerate(charts):
        chart_title = chart.replace("_", " ").title().replace("01 ", "").replace("02 ", "")
        slides.append(r"""
\begin{frame}[t]{%s}
\begin{center}
\includegraphics[width=0.85\textwidth]{%s/chart.pdf}
\end{center}
\bottomnote{Key concept for financial data analysis}
\end{frame}
""" % (chart_title, chart))

    # Summary slide
    slides.append(r"""
\begin{frame}[t]{Lesson Summary}
\textbf{Key Takeaways:}
\begin{itemize}
%s
\end{itemize}

\vspace{1em}
\textbf{Practice:} Apply these concepts to the stock price dataset.
\end{frame}

\end{document}
""" % obj_items)

    return PREAMBLE + "\n".join(slides)

def main():
    for folder, (title, objectives, charts) in LESSONS.items():
        content = generate_lesson_tex(folder, title, objectives, charts)
        filepath = BASE_DIR / folder / f"{folder}.tex"
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Created: {filepath}")

    print("\nAll L07-L12 .tex files generated!")

if __name__ == '__main__':
    main()
