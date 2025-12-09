"""Generate .tex files for L02-L06"""
from pathlib import Path

BASE_DIR = Path(__file__).parent

PREAMBLE = r"""\documentclass[8pt,aspectratio=169]{beamer}
\usetheme{Madrid}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{adjustbox}
\usepackage{multicol}
\usepackage{amsmath}
\usepackage{amssymb}

% Color definitions
\definecolor{mlblue}{RGB}{0,102,204}
\definecolor{mlpurple}{RGB}{51,51,178}
\definecolor{mllavender}{RGB}{173,173,224}
\definecolor{mllavender2}{RGB}{193,193,232}
\definecolor{mllavender3}{RGB}{204,204,235}
\definecolor{mllavender4}{RGB}{214,214,239}
\definecolor{mlorange}{RGB}{255, 127, 14}
\definecolor{mlgreen}{RGB}{44, 160, 44}
\definecolor{mlred}{RGB}{214, 39, 40}

\setbeamercolor{palette primary}{bg=mllavender3,fg=mlpurple}
\setbeamercolor{palette secondary}{bg=mllavender2,fg=mlpurple}
\setbeamercolor{palette tertiary}{bg=mllavender,fg=white}
\setbeamercolor{palette quaternary}{bg=mlpurple,fg=white}
\setbeamercolor{structure}{fg=mlpurple}
\setbeamercolor{frametitle}{fg=mlpurple,bg=mllavender3}
\setbeamercolor{block title}{bg=mllavender2,fg=mlpurple}
\setbeamercolor{block body}{bg=mllavender4,fg=black}

\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{itemize items}[circle]
\setbeamersize{text margin left=5mm,text margin right=5mm}

\newcommand{\bottomnote}[1]{%
\vfill\vspace{-2mm}
\textcolor{mllavender2}{\rule{\textwidth}{0.4pt}}
\vspace{1mm}\footnotesize\textbf{#1}}

"""

# L02: Data Structures
L02_CONTENT = r"""
\title{Lesson 02: Data Structures}
\subtitle{Data Science with Python -- BSc Course}
\author{Data Science Program}
\date{45 Minutes}

\begin{document}

\begin{frame}[plain]
\titlepage
\end{frame}

\begin{frame}[t]{Learning Objectives}
\textbf{After this lesson, you will be able to:}
\begin{itemize}
\item Create and manipulate Python lists
\item Access elements using indexing and slicing
\item Build dictionaries for key-value data storage
\item Apply list comprehensions for efficient data processing
\end{itemize}

\vspace{1em}
\textbf{Finance Application:} Store portfolio holdings as lists and dictionaries.

\bottomnote{Data structures are containers for organizing information}
\end{frame}

\begin{frame}[t]{List Indexing}
\begin{columns}[T]
\column{0.55\textwidth}
\begin{center}
\includegraphics[width=\textwidth]{01_list_indexing/chart.pdf}
\end{center}

\column{0.42\textwidth}
\textbf{Creating Lists:}\\
\texttt{prices = [185, 190, 188, 195]}

\vspace{0.5em}
\textbf{Accessing Elements:}\\
\texttt{prices[0]} $\rightarrow$ \texttt{185} (first)\\
\texttt{prices[-1]} $\rightarrow$ \texttt{195} (last)\\
\texttt{prices[1]} $\rightarrow$ \texttt{190} (second)

\vspace{0.5em}
\textbf{Remember:}\\
Python indexing starts at 0!
\end{columns}

\bottomnote{Negative indices count from the end: -1 is last element}
\end{frame}

\begin{frame}[t]{Slicing Notation}
\begin{columns}[T]
\column{0.48\textwidth}
\begin{center}
\includegraphics[width=\textwidth]{02_slicing_notation/chart.pdf}
\end{center}

\column{0.48\textwidth}
\textbf{Slice Syntax:} \texttt{list[start:end:step]}

\vspace{0.5em}
\texttt{prices = [185, 190, 188, 195, 182]}

\vspace{0.3em}
\texttt{prices[1:4]} $\rightarrow$ \texttt{[190, 188, 195]}\\
\texttt{prices[:3]} $\rightarrow$ \texttt{[185, 190, 188]}\\
\texttt{prices[2:]} $\rightarrow$ \texttt{[188, 195, 182]}\\
\texttt{prices[::2]} $\rightarrow$ \texttt{[185, 188, 182]}

\vspace{0.5em}
\textbf{Key:} End index is exclusive
\end{columns}

\bottomnote{Slicing creates a new list -- original unchanged}
\end{frame}

\begin{frame}[t]{Dictionary Structure}
\begin{columns}[T]
\column{0.48\textwidth}
\begin{center}
\includegraphics[width=\textwidth]{03_dictionary_structure/chart.pdf}
\end{center}

\column{0.48\textwidth}
\textbf{Key-Value Pairs:}

\vspace{0.3em}
\texttt{portfolio = \{}\\
\texttt{~~~~"AAPL": 50,}\\
\texttt{~~~~"MSFT": 30,}\\
\texttt{~~~~"GOOGL": 20}\\
\texttt{\}}

\vspace{0.5em}
\textbf{Access by Key:}\\
\texttt{portfolio["AAPL"]} $\rightarrow$ \texttt{50}

\vspace{0.5em}
\texttt{portfolio.keys()}\\
\texttt{portfolio.values()}
\end{columns}

\bottomnote{Dictionaries provide O(1) lookup -- very fast access}
\end{frame}

\begin{frame}[t]{Nested Data Structures}
\begin{center}
\includegraphics[width=0.7\textwidth]{04_nested_structures/chart.pdf}
\end{center}

\textbf{Real-World Portfolio Structure:}\\
\texttt{holdings = \{"AAPL": \{"shares": 50, "price": 185.5\}, ....\}}

\bottomnote{Nested structures model complex financial data}
\end{frame}

\begin{frame}[t]{List Methods}
\begin{columns}[T]
\column{0.55\textwidth}
\begin{center}
\includegraphics[width=\textwidth]{05_list_methods/chart.pdf}
\end{center}

\column{0.42\textwidth}
\textbf{Adding Elements:}\\
\texttt{prices.append(200)}\\
\texttt{prices.insert(0, 180)}

\vspace{0.5em}
\textbf{Removing:}\\
\texttt{prices.remove(188)}\\
\texttt{prices.pop()} -- removes last

\vspace{0.5em}
\textbf{Sorting:}\\
\texttt{prices.sort()}\\
\texttt{prices.reverse()}
\end{columns}

\bottomnote{Methods modify the list in-place (except sorted())}
\end{frame}

\begin{frame}[t]{Portfolio as Dictionary}
\begin{columns}[T]
\column{0.48\textwidth}
\begin{center}
\includegraphics[width=\textwidth]{06_portfolio_dict/chart.pdf}
\end{center}

\column{0.48\textwidth}
\textbf{Portfolio Dictionary:}

\texttt{prices = \{}\\
\texttt{~~~~"AAPL": 185.50,}\\
\texttt{~~~~"MSFT": 378.20,}\\
\texttt{~~~~"GOOGL": 141.80}\\
\texttt{\}}

\vspace{0.5em}
\textbf{Calculate Total:}\\
\texttt{total = sum(prices.values())}

\vspace{0.5em}
\textbf{Check Existence:}\\
\texttt{"AAPL" in prices} $\rightarrow$ \texttt{True}
\end{columns}

\bottomnote{Dictionaries are ideal for ticker-to-data mappings}
\end{frame}

\begin{frame}[t]{List Comprehension}
\begin{columns}[T]
\column{0.48\textwidth}
\begin{center}
\includegraphics[width=\textwidth]{07_list_comprehension/chart.pdf}
\end{center}

\column{0.48\textwidth}
\textbf{Traditional Loop:}\\
\texttt{returns = []}\\
\texttt{for p in prices:}\\
\texttt{~~~~returns.append(p * 1.05)}

\vspace{0.5em}
\textbf{List Comprehension:}\\
\texttt{returns = [p * 1.05 for p in prices]}

\vspace{0.5em}
\textbf{With Condition:}\\
\texttt{high = [p for p in prices if p > 190]}
\end{columns}

\bottomnote{Comprehensions are more Pythonic and often faster}
\end{frame}

\begin{frame}[t]{Choosing the Right Structure}
\begin{center}
\includegraphics[width=0.85\textwidth]{08_structure_selection/chart.pdf}
\end{center}

\bottomnote{Lists for sequences, dicts for lookups, sets for uniqueness}
\end{frame}

\begin{frame}[t]{Hands-on Exercise (25 min)}
\textbf{Build a portfolio tracker:}

\begin{enumerate}
\item Create a list of stock tickers:\\
\texttt{tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]}

\item Create a dictionary with shares owned:\\
\texttt{shares = \{"AAPL": 50, "MSFT": 30, ...\}}

\item Create a dictionary with current prices

\item Calculate portfolio value using list comprehension:\\
\texttt{values = [shares[t] * prices[t] for t in tickers]}

\item Find total portfolio value: \texttt{sum(values)}

\item Filter stocks worth more than \$5000
\end{enumerate}

\bottomnote{Save your work -- we'll add more features next lesson}
\end{frame}

\begin{frame}[t]{Lesson Summary}
\textbf{Key Takeaways:}
\begin{itemize}
\item Lists store ordered sequences (accessed by index)
\item Dictionaries store key-value pairs (accessed by key)
\item Slicing extracts portions: \texttt{list[start:end]}
\item List comprehensions create lists efficiently
\item Choose structure based on access pattern
\end{itemize}

\vspace{1em}
\textbf{Next Lesson:} Control Flow (if/else, loops)

\bottomnote{Data structures + control flow = programming logic}
\end{frame}

\end{document}
"""

# L03: Control Flow
L03_CONTENT = r"""
\title{Lesson 03: Control Flow}
\subtitle{Data Science with Python -- BSc Course}
\author{Data Science Program}
\date{45 Minutes}

\begin{document}

\begin{frame}[plain]
\titlepage
\end{frame}

\begin{frame}[t]{Learning Objectives}
\textbf{After this lesson, you will be able to:}
\begin{itemize}
\item Write conditional statements with if/elif/else
\item Create loops to iterate over data
\item Implement trading rules using control flow
\item Use break and continue for loop control
\end{itemize}

\vspace{1em}
\textbf{Finance Application:} Implement trading signals and position sizing rules.

\bottomnote{Control flow determines which code executes based on conditions}
\end{frame}

\begin{frame}[t]{If-Else Statements}
\begin{columns}[T]
\column{0.48\textwidth}
\begin{center}
\includegraphics[width=\textwidth]{01_if_else_flowchart/chart.pdf}
\end{center}

\column{0.48\textwidth}
\textbf{Basic Structure:}

\texttt{if price > 200:}\\
\texttt{~~~~signal = "SELL"}\\
\texttt{elif price < 150:}\\
\texttt{~~~~signal = "BUY"}\\
\texttt{else:}\\
\texttt{~~~~signal = "HOLD"}

\vspace{0.5em}
\textbf{Key Points:}
\begin{itemize}
\item Colon after condition
\item Indentation matters!
\item elif is optional
\end{itemize}
\end{columns}

\bottomnote{Indentation (4 spaces) defines code blocks in Python}
\end{frame}

\begin{frame}[t]{For Loops}
\begin{columns}[T]
\column{0.55\textwidth}
\begin{center}
\includegraphics[width=\textwidth]{02_for_loop_visual/chart.pdf}
\end{center}

\column{0.42\textwidth}
\textbf{Iterate Over List:}\\
\texttt{for price in prices:}\\
\texttt{~~~~print(price)}

\vspace{0.5em}
\textbf{With Index:}\\
\texttt{for i, p in enumerate(prices):}\\
\texttt{~~~~print(f"Day \{i\}: \{p\}")}

\vspace{0.5em}
\textbf{Range:}\\
\texttt{for i in range(5):}\\
\texttt{~~~~print(i)}  \# 0,1,2,3,4
\end{columns}

\bottomnote{For loops iterate a known number of times}
\end{frame}

\begin{frame}[t]{While Loops}
\begin{columns}[T]
\column{0.48\textwidth}
\begin{center}
\includegraphics[width=\textwidth]{03_while_loop_diagram/chart.pdf}
\end{center}

\column{0.48\textwidth}
\textbf{Basic While:}\\
\texttt{balance = 10000}\\
\texttt{while balance > 5000:}\\
\texttt{~~~~balance *= 0.95}\\
\texttt{~~~~print(balance)}

\vspace{0.5em}
\textbf{Use Cases:}
\begin{itemize}
\item Unknown iterations
\item Waiting for condition
\item Simulation until target
\end{itemize}

\vspace{0.3em}
\textbf{Warning:} Infinite loops!
\end{columns}

\bottomnote{While loops continue until condition becomes False}
\end{frame}

\begin{frame}[t]{Nested Loops}
\begin{center}
\includegraphics[width=0.65\textwidth]{04_nested_loops/chart.pdf}
\end{center}

\textbf{Example:} Analyze multiple stocks across multiple days

\texttt{for ticker in ["AAPL", "MSFT"]:}\\
\texttt{~~~~for day in range(5):}\\
\texttt{~~~~~~~~print(f"\{ticker\} day \{day\}")}

\bottomnote{Nested loops multiply iterations -- watch performance}
\end{frame}

\begin{frame}[t]{Break and Continue}
\begin{columns}[T]
\column{0.55\textwidth}
\begin{center}
\includegraphics[width=\textwidth]{05_break_continue/chart.pdf}
\end{center}

\column{0.42\textwidth}
\textbf{Break:} Exit loop entirely

\texttt{for price in prices:}\\
\texttt{~~~~if price > 200:}\\
\texttt{~~~~~~~~break}  \# stop now

\vspace{0.5em}
\textbf{Continue:} Skip to next iteration

\texttt{for price in prices:}\\
\texttt{~~~~if price < 0:}\\
\texttt{~~~~~~~~continue}  \# skip this\\
\texttt{~~~~process(price)}
\end{columns}

\bottomnote{Break stops loop; continue skips current iteration}
\end{frame}

\begin{frame}[t]{Trading Rules Decision Tree}
\begin{center}
\includegraphics[width=0.75\textwidth]{06_trading_rules/chart.pdf}
\end{center}

\bottomnote{Trading rules are nested conditionals in code}
\end{frame}

\begin{frame}[t]{Loop Comparison}
\begin{center}
\includegraphics[width=0.85\textwidth]{07_loop_comparison/chart.pdf}
\end{center}

\bottomnote{Choose loop type based on whether iterations are known}
\end{frame}

\begin{frame}[t]{Control Flow Patterns}
\begin{center}
\includegraphics[width=0.8\textwidth]{08_control_patterns/chart.pdf}
\end{center}

\bottomnote{Combine patterns for complex trading logic}
\end{frame}

\begin{frame}[t]{Hands-on Exercise (25 min)}
\textbf{Implement a simple trading system:}

\begin{enumerate}
\item Create price list: \texttt{prices = [180, 185, 195, 188, 205, 198]}

\item Implement trading rules:
\begin{itemize}
\item If price > 200: SELL
\item If price < 185: BUY
\item Else: HOLD
\end{itemize}

\item Loop through prices and generate signals

\item Count total BUY, SELL, HOLD signals

\item Find first price that triggers SELL (use break)

\item Skip negative prices if any (use continue)
\end{enumerate}

\bottomnote{This forms the basis of algorithmic trading}
\end{frame}

\begin{frame}[t]{Lesson Summary}
\textbf{Key Takeaways:}
\begin{itemize}
\item if/elif/else for conditional execution
\item for loops iterate over sequences
\item while loops continue until condition is false
\item break exits loop, continue skips iteration
\item Indentation defines code blocks
\end{itemize}

\vspace{1em}
\textbf{Next Lesson:} Functions

\bottomnote{Control flow + functions = modular trading systems}
\end{frame}

\end{document}
"""

# L04: Functions
L04_CONTENT = r"""
\title{Lesson 04: Functions}
\subtitle{Data Science with Python -- BSc Course}
\author{Data Science Program}
\date{45 Minutes}

\begin{document}

\begin{frame}[plain]
\titlepage
\end{frame}

\begin{frame}[t]{Learning Objectives}
\textbf{After this lesson, you will be able to:}
\begin{itemize}
\item Define and call functions with parameters
\item Use return statements to output values
\item Understand variable scope (local vs global)
\item Write docstrings for documentation
\end{itemize}

\vspace{1em}
\textbf{Finance Application:} Create reusable functions for return calculations and risk metrics.

\bottomnote{Functions are building blocks of modular code}
\end{frame}

\begin{frame}[t]{Function Anatomy}
\begin{center}
\includegraphics[width=0.75\textwidth]{01_function_anatomy/chart.pdf}
\end{center}

\bottomnote{def keyword, name, parameters, colon, indented body, return}
\end{frame}

\begin{frame}[t]{Parameter Passing}
\begin{center}
\includegraphics[width=0.8\textwidth]{02_parameter_passing/chart.pdf}
\end{center}

\bottomnote{Python passes object references -- mutable objects can change}
\end{frame}

\begin{frame}[t]{Return Values}
\begin{center}
\includegraphics[width=0.75\textwidth]{03_return_flowchart/chart.pdf}
\end{center}

\bottomnote{Functions without return statement return None}
\end{frame}

\begin{frame}[t]{Variable Scope}
\begin{center}
\includegraphics[width=0.8\textwidth]{04_scope_diagram/chart.pdf}
\end{center}

\bottomnote{Local variables are destroyed when function ends}
\end{frame}

\begin{frame}[t]{Docstrings}
\begin{center}
\includegraphics[width=0.85\textwidth]{05_docstring_format/chart.pdf}
\end{center}

\bottomnote{Access docstring with help(function) or function.\_\_doc\_\_}
\end{frame}

\begin{frame}[t]{Function Call Stack}
\begin{center}
\includegraphics[width=0.7\textwidth]{06_call_stack/chart.pdf}
\end{center}

\bottomnote{Stack grows with nested calls, shrinks as functions return}
\end{frame}

\begin{frame}[t]{Pure vs Impure Functions}
\begin{center}
\includegraphics[width=0.85\textwidth]{07_pure_vs_impure/chart.pdf}
\end{center}

\bottomnote{Pure functions are easier to test and debug}
\end{frame}

\begin{frame}[t]{Finance Functions Library}
\begin{center}
\includegraphics[width=0.85\textwidth]{08_finance_functions/chart.pdf}
\end{center}

\bottomnote{Build your own library of financial calculation functions}
\end{frame}

\begin{frame}[t]{Hands-on Exercise (25 min)}
\textbf{Build a finance functions library:}

\begin{enumerate}
\item \texttt{calculate\_return(buy, sell)} -- percentage return

\item \texttt{annualize\_return(daily\_ret, days=252)} -- annualized

\item \texttt{calculate\_volatility(returns)} -- standard deviation

\item \texttt{sharpe\_ratio(returns, rf=0.02)} -- risk-adjusted return

\item Test each function with sample data

\item Add docstrings to all functions
\end{enumerate}

\bottomnote{These functions will be used throughout the course}
\end{frame}

\begin{frame}[t]{Lesson Summary}
\textbf{Key Takeaways:}
\begin{itemize}
\item Functions encapsulate reusable logic
\item Parameters pass data in, return sends data out
\item Local scope: variables exist only inside function
\item Docstrings document function purpose and usage
\item Pure functions are predictable and testable
\end{itemize}

\vspace{1em}
\textbf{Next Lesson:} DataFrames Introduction

\bottomnote{Functions + pandas = powerful financial analysis}
\end{frame}

\end{document}
"""

# L05: DataFrames Introduction
L05_CONTENT = r"""
\title{Lesson 05: DataFrames Introduction}
\subtitle{Data Science with Python -- BSc Course}
\author{Data Science Program}
\date{45 Minutes}

\begin{document}

\begin{frame}[plain]
\titlepage
\end{frame}

\begin{frame}[t]{Learning Objectives}
\textbf{After this lesson, you will be able to:}
\begin{itemize}
\item Import pandas and create DataFrames
\item Load data from CSV files
\item Explore data with head(), tail(), info(), describe()
\item Understand DataFrame structure (index, columns, values)
\end{itemize}

\vspace{1em}
\textbf{Finance Application:} Load and explore stock price data.

\bottomnote{pandas is THE library for data manipulation in Python}
\end{frame}

\begin{frame}[t]{DataFrame Structure}
\begin{center}
\includegraphics[width=0.8\textwidth]{01_dataframe_structure/chart.pdf}
\end{center}

\bottomnote{DataFrames are 2D labeled data structures}
\end{frame}

\begin{frame}[t]{Series vs DataFrame}
\begin{center}
\includegraphics[width=0.85\textwidth]{02_series_vs_dataframe/chart.pdf}
\end{center}

\bottomnote{Series = 1 column; DataFrame = multiple Series}
\end{frame}

\begin{frame}[t]{Loading CSV Data}
\begin{center}
\includegraphics[width=0.8\textwidth]{03_csv_loading/chart.pdf}
\end{center}

\bottomnote{pd.read\_csv() handles most CSV formats automatically}
\end{frame}

\begin{frame}[t]{Viewing Data: head() and tail()}
\begin{center}
\includegraphics[width=0.85\textwidth]{04_head_tail/chart.pdf}
\end{center}

\bottomnote{Always inspect data after loading!}
\end{frame}

\begin{frame}[t]{DataFrame Info}
\begin{center}
\includegraphics[width=0.85\textwidth]{05_info_breakdown/chart.pdf}
\end{center}

\bottomnote{info() reveals data types and missing values}
\end{frame}

\begin{frame}[t]{Summary Statistics}
\begin{center}
\includegraphics[width=0.85\textwidth]{06_describe_stats/chart.pdf}
\end{center}

\bottomnote{describe() provides statistical summary of numeric columns}
\end{frame}

\begin{frame}[t]{Index and Columns}
\begin{center}
\includegraphics[width=0.8\textwidth]{07_index_columns/chart.pdf}
\end{center}

\bottomnote{Index labels rows; columns label data fields}
\end{frame}

\begin{frame}[t]{Stock Data Example}
\begin{center}
\includegraphics[width=0.9\textwidth]{08_stock_example/chart.pdf}
\end{center}

\bottomnote{Financial time series are natural DataFrames}
\end{frame}

\begin{frame}[t]{Hands-on Exercise (25 min)}
\textbf{Explore stock price data:}

\begin{enumerate}
\item Load the stock data:\\
\texttt{df = pd.read\_csv("../datasets/stock\_prices.csv")}

\item View first 10 rows: \texttt{df.head(10)}

\item Check data types: \texttt{df.info()}

\item Get statistics: \texttt{df.describe()}

\item Access column names: \texttt{df.columns}

\item Check shape: \texttt{df.shape}

\item Find which stock has highest mean price
\end{enumerate}

\bottomnote{Exploration before analysis prevents costly mistakes}
\end{frame}

\begin{frame}[t]{Lesson Summary}
\textbf{Key Takeaways:}
\begin{itemize}
\item pandas DataFrame is the core data structure
\item pd.read\_csv() loads CSV files easily
\item head()/tail() show first/last rows
\item info() shows data types and missing values
\item describe() provides statistical summary
\end{itemize}

\vspace{1em}
\textbf{Next Lesson:} Selection and Filtering

\bottomnote{Loading data is step 1 -- now we'll learn to slice it}
\end{frame}

\end{document}
"""

# L06: Selection and Filtering
L06_CONTENT = r"""
\title{Lesson 06: Selection and Filtering}
\subtitle{Data Science with Python -- BSc Course}
\author{Data Science Program}
\date{45 Minutes}

\begin{document}

\begin{frame}[plain]
\titlepage
\end{frame}

\begin{frame}[t]{Learning Objectives}
\textbf{After this lesson, you will be able to:}
\begin{itemize}
\item Select columns using bracket and dot notation
\item Access rows with iloc (position) and loc (label)
\item Filter data using boolean conditions
\item Combine multiple conditions with \& and |
\end{itemize}

\vspace{1em}
\textbf{Finance Application:} Screen stocks by price, volume, and other criteria.

\bottomnote{Selection and filtering extract relevant data for analysis}
\end{frame}

\begin{frame}[t]{Column Selection}
\begin{center}
\includegraphics[width=0.85\textwidth]{01_column_selection/chart.pdf}
\end{center}

\bottomnote{Single brackets return Series; double brackets return DataFrame}
\end{frame}

\begin{frame}[t]{iloc vs loc}
\begin{center}
\includegraphics[width=0.85\textwidth]{02_iloc_vs_loc/chart.pdf}
\end{center}

\bottomnote{iloc: integer position; loc: label-based}
\end{frame}

\begin{frame}[t]{Boolean Masking}
\begin{center}
\includegraphics[width=0.85\textwidth]{03_boolean_mask/chart.pdf}
\end{center}

\bottomnote{Boolean mask selects rows where condition is True}
\end{frame}

\begin{frame}[t]{Conditional Filtering}
\begin{center}
\includegraphics[width=0.85\textwidth]{04_conditional_filtering/chart.pdf}
\end{center}

\bottomnote{Filtering creates a new DataFrame -- original unchanged}
\end{frame}

\begin{frame}[t]{Multiple Conditions}
\begin{center}
\includegraphics[width=0.85\textwidth]{05_multiple_conditions/chart.pdf}
\end{center}

\bottomnote{Use parentheses around each condition!}
\end{frame}

\begin{frame}[t]{Chained Filtering}
\begin{center}
\includegraphics[width=0.85\textwidth]{06_chained_filtering/chart.pdf}
\end{center}

\bottomnote{query() method is more readable for complex filters}
\end{frame}

\begin{frame}[t]{Selection Methods Comparison}
\begin{center}
\includegraphics[width=0.85\textwidth]{07_selection_comparison/chart.pdf}
\end{center}

\bottomnote{Choose method based on what you need to select}
\end{frame}

\begin{frame}[t]{Stock Screening Workflow}
\begin{center}
\includegraphics[width=0.85\textwidth]{08_stock_screening/chart.pdf}
\end{center}

\bottomnote{Stock screening = loading + filtering + selecting + sorting}
\end{frame}

\begin{frame}[t]{Hands-on Exercise (25 min)}
\textbf{Build a stock screener:}

\begin{enumerate}
\item Load stock data from CSV

\item Select only AAPL and MSFT columns

\item Filter rows where AAPL > 185

\item Filter rows where AAPL > 185 AND MSFT > 375

\item Use query() for the same filter

\item Select first 10 trading days using iloc

\item Sort by AAPL price descending
\end{enumerate}

\bottomnote{Stock screeners are fundamental tools in finance}
\end{frame}

\begin{frame}[t]{Lesson Summary}
\textbf{Key Takeaways:}
\begin{itemize}
\item df["col"] selects single column as Series
\item iloc uses integer positions; loc uses labels
\item Boolean conditions create True/False masks
\item Combine conditions with \& (and) and | (or)
\item query() is cleaner for complex filters
\end{itemize}

\vspace{1em}
\textbf{Next Lesson:} Missing Data and Cleaning

\bottomnote{Week 1 complete! You can now load, explore, and filter data}
\end{frame}

\end{document}
"""

def generate_tex_files():
    """Generate all .tex files for L02-L06"""

    lessons = [
        ("L02_Data_Structures", L02_CONTENT),
        ("L03_Control_Flow", L03_CONTENT),
        ("L04_Functions", L04_CONTENT),
        ("L05_DataFrames_Introduction", L05_CONTENT),
        ("L06_Selection_Filtering", L06_CONTENT),
    ]

    for folder, content in lessons:
        filepath = BASE_DIR / folder / f"{folder}.tex"
        full_content = PREAMBLE + content

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_content)

        print(f"Created: {filepath}")

    print("\nAll .tex files generated!")

if __name__ == '__main__':
    generate_tex_files()
