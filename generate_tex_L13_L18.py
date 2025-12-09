"""Generate .tex files for L13-L18"""
from pathlib import Path

BASE_DIR = Path(__file__).parent

PREAMBLE = r"""\documentclass[8pt,aspectratio=169]{beamer}
\usetheme{Madrid}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}

\definecolor{mlpurple}{RGB}{51,51,178}
\definecolor{mllavender}{RGB}{173,173,224}
\definecolor{mllavender3}{RGB}{204,204,235}

\setbeamercolor{palette primary}{bg=mllavender3,fg=mlpurple}
\setbeamercolor{structure}{fg=mlpurple}
\setbeamercolor{frametitle}{fg=mlpurple,bg=mllavender3}

\setbeamertemplate{navigation symbols}{}
\setbeamersize{text margin left=5mm,text margin right=5mm}

\newcommand{\bottomnote}[1]{\vfill\footnotesize\textbf{#1}}

"""

LESSONS = {
    "L13_Descriptive_Statistics": (
        "Lesson 13: Descriptive Statistics",
        ["Calculate mean, median, mode", "Measure dispersion (std, variance, range)",
         "Interpret quartiles and percentiles", "Analyze skewness and kurtosis"],
        ["01_central_tendency", "02_dispersion", "03_quartiles", "04_skewness",
         "05_kurtosis", "06_summary_table", "07_finance_stats", "08_comparison"]
    ),
    "L14_Distributions": (
        "Lesson 14: Probability Distributions",
        ["Understand normal distribution properties", "Apply distributions to stock returns",
         "Use QQ-plots for normality testing", "Recognize fat tails in finance"],
        ["01_normal_distribution", "02_binomial", "03_pdf_cdf", "04_stock_returns",
         "05_qq_plot", "06_fat_tails", "07_distribution_fitting", "08_finance_distributions"]
    ),
    "L15_Hypothesis_Testing": (
        "Lesson 15: Hypothesis Testing",
        ["Formulate null and alternative hypotheses", "Perform t-tests",
         "Interpret p-values correctly", "Construct confidence intervals"],
        ["01_hypothesis_concept", "02_null_alternative", "03_t_test", "04_p_value",
         "05_confidence_interval", "06_type_errors", "07_ab_testing", "08_finance_tests"]
    ),
    "L16_Correlation": (
        "Lesson 16: Correlation Analysis",
        ["Calculate Pearson and Spearman correlation", "Create correlation heatmaps",
         "Distinguish correlation from causation", "Apply to portfolio analysis"],
        ["01_correlation_scatter", "02_correlation_values", "03_pearson_spearman", "04_heatmap",
         "05_spurious_correlation", "06_causation", "07_rolling_correlation", "08_portfolio_correlation"]
    ),
    "L17_Matplotlib_Basics": (
        "Lesson 17: Matplotlib Basics",
        ["Create line, bar, and scatter plots", "Customize colors, labels, legends",
         "Build multi-panel figures", "Add annotations and formatting"],
        ["01_line_plot", "02_bar_chart", "03_histogram", "04_scatter_plot",
         "05_subplots", "06_customization", "07_annotations", "08_finance_charts"]
    ),
    "L18_Seaborn_Plots": (
        "Lesson 18: Seaborn Statistical Plots",
        ["Use seaborn for statistical visualization", "Create distribution and categorical plots",
         "Build correlation heatmaps", "Apply professional styling"],
        ["01_seaborn_intro", "02_distribution_plots", "03_categorical_plots", "04_regression_plots",
         "05_heatmaps", "06_pairplot", "07_styling", "08_finance_seaborn"]
    )
}

def generate_lesson_tex(folder, title, objectives, charts):
    slides = []
    slides.append(r"""
\title{%s}
\subtitle{Data Science with Python -- BSc Course}
\date{45 Minutes}

\begin{document}

\begin{frame}[plain]
\titlepage
\end{frame}

\begin{frame}[t]{Learning Objectives}
\textbf{After this lesson, you will be able to:}
\begin{itemize}
%s
\end{itemize}
\bottomnote{Finance application: Statistical analysis of market data}
\end{frame}
""" % (title, "\n".join([f"\\item {obj}" for obj in objectives])))

    for chart in charts:
        chart_title = chart.replace("_", " ").title()
        for prefix in ['01 ', '02 ', '03 ', '04 ', '05 ', '06 ', '07 ', '08 ']:
            chart_title = chart_title.replace(prefix, '')
        slides.append(r"""
\begin{frame}[t]{%s}
\begin{center}
\includegraphics[width=0.85\textwidth]{%s/chart.pdf}
\end{center}
\bottomnote{Statistical foundation for data-driven decisions}
\end{frame}
""" % (chart_title, chart))

    slides.append(r"""
\begin{frame}[t]{Lesson Summary}
\textbf{Key Takeaways:}
\begin{itemize}
%s
\end{itemize}
\bottomnote{Statistics + Visualization = Data Science foundation}
\end{frame}

\end{document}
""" % "\n".join([f"\\item {obj}" for obj in objectives]))

    return PREAMBLE + "\n".join(slides)

def main():
    for folder, (title, objectives, charts) in LESSONS.items():
        content = generate_lesson_tex(folder, title, objectives, charts)
        filepath = BASE_DIR / folder / f"{folder}.tex"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created: {filepath}")
    print("\nAll L13-L18 .tex files generated!")

if __name__ == '__main__':
    main()
