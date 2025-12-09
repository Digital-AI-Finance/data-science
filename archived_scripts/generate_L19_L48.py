"""Generate charts and .tex files for L19-L48: ML, DL, NLP, Deployment"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

MLPURPLE = '#3333B2'
MLLAVENDER = '#ADADE0'
MLBLUE = '#0066CC'
MLORANGE = '#FF7F0E'
MLGREEN = '#2CA02C'
MLRED = '#D42728'

plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'

BASE_DIR = Path(__file__).parent

def save_chart(fig, lesson_folder, chart_name):
    chart_dir = BASE_DIR / lesson_folder / chart_name
    chart_dir.mkdir(parents=True, exist_ok=True)
    output_path = chart_dir / 'chart.pdf'
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)

def create_chart(folder, name, chart_type='generic'):
    fig, ax = plt.subplots(figsize=(10, 6))
    np.random.seed(hash(name) % 2**32)

    title = name.replace('_', ' ').title()
    for i in range(1, 9):
        title = title.replace(f'{i:02d} ', '')

    if 'regression' in name.lower() or 'linear' in name.lower():
        x = np.linspace(0, 10, 50)
        y = 2*x + 1 + np.random.randn(50)*2
        ax.scatter(x, y, alpha=0.6, color=MLPURPLE)
        ax.plot(x, 2*x + 1, color=MLRED, linewidth=2, label='Fit')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
    elif 'classification' in name.lower() or 'decision' in name.lower():
        x1 = np.random.randn(50) + 2
        y1 = np.random.randn(50) + 2
        x2 = np.random.randn(50) - 1
        y2 = np.random.randn(50) - 1
        ax.scatter(x1, y1, c=MLBLUE, label='Class A', alpha=0.7)
        ax.scatter(x2, y2, c=MLORANGE, label='Class B', alpha=0.7)
        ax.legend()
    elif 'confusion' in name.lower() or 'matrix' in name.lower():
        data = np.array([[45, 5], [8, 42]])
        im = ax.imshow(data, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred 0', 'Pred 1'])
        ax.set_yticklabels(['True 0', 'True 1'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(data[i, j]), ha='center', va='center', fontsize=14)
    elif 'roc' in name.lower() or 'auc' in name.lower():
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)
        ax.plot(fpr, tpr, color=MLPURPLE, linewidth=2, label='ROC (AUC=0.85)')
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        ax.fill_between(fpr, tpr, alpha=0.2, color=MLPURPLE)
    elif 'cluster' in name.lower() or 'kmeans' in name.lower():
        centers = [(2, 2), (-2, -2), (2, -2)]
        colors = [MLBLUE, MLORANGE, MLGREEN]
        for (cx, cy), c in zip(centers, colors):
            x = np.random.randn(30)*0.5 + cx
            y = np.random.randn(30)*0.5 + cy
            ax.scatter(x, y, c=c, alpha=0.7)
    elif 'neural' in name.lower() or 'network' in name.lower():
        ax.axis('off')
        layers = [3, 4, 4, 2]
        for l, n in enumerate(layers):
            for i in range(n):
                circle = plt.Circle((l*2, i - n/2 + 0.5), 0.3, color=MLPURPLE if l < len(layers)-1 else MLGREEN)
                ax.add_patch(circle)
                if l > 0:
                    for j in range(layers[l-1]):
                        ax.plot([(l-1)*2+0.3, l*2-0.3], [j - layers[l-1]/2 + 0.5, i - n/2 + 0.5],
                               color=MLLAVENDER, linewidth=0.5)
        ax.set_xlim(-1, 7)
        ax.set_ylim(-3, 3)
    elif 'loss' in name.lower() or 'training' in name.lower():
        epochs = np.arange(100)
        train_loss = 1 / (1 + epochs*0.1) + np.random.randn(100)*0.02
        val_loss = 1 / (1 + epochs*0.08) + np.random.randn(100)*0.03 + 0.1
        ax.plot(epochs, train_loss, label='Training', color=MLBLUE)
        ax.plot(epochs, val_loss, label='Validation', color=MLORANGE)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
    elif 'sentiment' in name.lower() or 'nlp' in name.lower():
        labels = ['Positive', 'Negative', 'Neutral']
        values = [45, 25, 30]
        ax.bar(labels, values, color=[MLGREEN, MLRED, MLBLUE])
        ax.set_ylabel('Count')
    elif 'deploy' in name.lower() or 'api' in name.lower():
        ax.axis('off')
        boxes = [(1, 3, 'Model'), (4, 3, 'API'), (7, 3, 'Client')]
        for x, y, label in boxes:
            rect = patches.FancyBboxPatch((x, y), 2, 1.5, boxstyle="round,pad=0.1",
                                           facecolor=MLLAVENDER, edgecolor=MLPURPLE, linewidth=2)
            ax.add_patch(rect)
            ax.text(x+1, y+0.75, label, ha='center', va='center', fontsize=12)
        ax.arrow(3, 3.75, 0.8, 0, head_width=0.2, color=MLPURPLE)
        ax.arrow(6, 3.75, 0.8, 0, head_width=0.2, color=MLPURPLE)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
    else:
        x = np.arange(10)
        y = np.random.rand(10)
        ax.bar(x, y, color=MLLAVENDER, edgecolor=MLPURPLE)

    ax.set_title(title, fontsize=14, fontweight='bold', color=MLPURPLE)
    save_chart(fig, folder, name)

# All lessons L19-L48
ALL_LESSONS = {
    # Agent 4: L19-L24 (Viz + Regression)
    "L19_Multi_Panel_Figures": (
        "Lesson 19: Multi-Panel Figures",
        ["Create subplots with plt.subplots()", "Arrange multiple visualizations",
         "Share axes and legends", "Build financial dashboards"],
        ["01_subplots_grid", "02_shared_axes", "03_mixed_layouts", "04_gridspec",
         "05_nested_plots", "06_figure_sizing", "07_dashboard_layout", "08_finance_dashboard"]
    ),
    "L20_Data_Storytelling": (
        "Lesson 20: Data Storytelling",
        ["Design visualizations for communication", "Apply color theory",
         "Create narrative flow", "Present to stakeholders"],
        ["01_storytelling_flow", "02_color_theory", "03_chart_selection", "04_annotations",
         "05_emphasis_techniques", "06_before_after", "07_executive_summary", "08_finance_story"]
    ),
    "L21_Linear_Regression": (
        "Lesson 21: Linear Regression",
        ["Understand OLS estimation", "Fit linear models with sklearn",
         "Interpret coefficients", "Apply to stock price prediction"],
        ["01_regression_concept", "02_ols_formula", "03_sklearn_api", "04_coefficient_interpretation",
         "05_prediction_line", "06_residuals", "07_assumptions", "08_capm_beta"]
    ),
    "L22_Regularization": (
        "Lesson 22: Regularization",
        ["Apply Ridge (L2) regularization", "Apply Lasso (L1) for feature selection",
         "Tune lambda with cross-validation", "Handle multicollinearity"],
        ["01_overfitting_visual", "02_ridge_concept", "03_lasso_concept", "04_coefficient_paths",
         "05_lambda_tuning", "06_cross_validation", "07_feature_selection", "08_finance_regularization"]
    ),
    "L23_Regression_Metrics": (
        "Lesson 23: Regression Metrics",
        ["Calculate MSE, RMSE, MAE", "Interpret R-squared",
         "Compare models fairly", "Handle time series validation"],
        ["01_mse_formula", "02_rmse_mae", "03_r_squared", "04_adjusted_r2",
         "05_residual_analysis", "06_model_comparison", "07_time_series_cv", "08_finance_metrics"]
    ),
    "L24_Factor_Models": (
        "Lesson 24: Factor Models",
        ["Build multi-factor regression", "Understand Fama-French factors",
         "Interpret factor loadings", "Create complete ML pipelines"],
        ["01_factor_concept", "02_fama_french", "03_factor_loadings", "04_alpha_beta",
         "05_multi_factor", "06_pipeline_sklearn", "07_model_persistence", "08_portfolio_factors"]
    ),

    # Agent 5: L25-L30 (Classification + Clustering)
    "L25_Logistic_Regression": (
        "Lesson 25: Logistic Regression",
        ["Understand sigmoid function", "Build binary classifiers",
         "Interpret odds ratios", "Predict market direction"],
        ["01_sigmoid_function", "02_decision_boundary", "03_sklearn_logistic", "04_odds_ratio",
         "05_multiclass", "06_regularization", "07_probability_output", "08_market_direction"]
    ),
    "L26_Decision_Trees": (
        "Lesson 26: Decision Trees",
        ["Build decision tree classifiers", "Understand splitting criteria",
         "Apply Random Forest ensemble", "Interpret feature importance"],
        ["01_tree_structure", "02_gini_entropy", "03_tree_sklearn", "04_overfitting",
         "05_random_forest", "06_feature_importance", "07_tree_visualization", "08_finance_trees"]
    ),
    "L27_Classification_Metrics": (
        "Lesson 27: Classification Metrics",
        ["Build confusion matrices", "Calculate precision and recall",
         "Plot ROC curves", "Handle class imbalance"],
        ["01_confusion_matrix", "02_accuracy_problems", "03_precision_recall", "04_f1_score",
         "05_roc_curve", "06_auc_interpretation", "07_threshold_tuning", "08_finance_classification"]
    ),
    "L28_Class_Imbalance": (
        "Lesson 28: Class Imbalance",
        ["Identify imbalanced datasets", "Apply SMOTE oversampling",
         "Use class weights", "Evaluate fairly"],
        ["01_imbalance_problem", "02_sampling_strategies", "03_smote", "04_class_weights",
         "05_stratified_cv", "06_precision_recall_tradeoff", "07_cost_sensitive", "08_fraud_detection"]
    ),
    "L29_KMeans_Clustering": (
        "Lesson 29: K-Means Clustering",
        ["Apply K-Means algorithm", "Choose optimal K (elbow method)",
         "Interpret cluster centers", "Segment financial assets"],
        ["01_kmeans_concept", "02_algorithm_steps", "03_sklearn_kmeans", "04_elbow_method",
         "05_silhouette_score", "06_cluster_visualization", "07_centroid_interpretation", "08_asset_clustering"]
    ),
    "L30_Hierarchical_Clustering": (
        "Lesson 30: Hierarchical Clustering",
        ["Build dendrograms", "Choose linkage methods",
         "Cut dendrograms for clusters", "Apply to portfolio construction"],
        ["01_hierarchical_concept", "02_linkage_methods", "03_dendrogram", "04_cutting_tree",
         "05_agglomerative", "06_correlation_clustering", "07_cluster_comparison", "08_portfolio_clustering"]
    ),

    # Agent 6: L31-L36 (PCA + Neural Networks)
    "L31_PCA": (
        "Lesson 31: PCA Dimensionality Reduction",
        ["Understand principal components", "Apply PCA with sklearn",
         "Interpret explained variance", "Reduce feature dimensions"],
        ["01_pca_concept", "02_eigenvalues", "03_sklearn_pca", "04_scree_plot",
         "05_explained_variance", "06_component_loadings", "07_visualization", "08_factor_extraction"]
    ),
    "L32_ML_Pipeline": (
        "Lesson 32: Complete ML Pipeline",
        ["Build sklearn pipelines", "Apply cross-validation properly",
         "Tune hyperparameters", "Prevent data leakage"],
        ["01_pipeline_concept", "02_sklearn_pipeline", "03_preprocessing_steps", "04_cross_validation",
         "05_grid_search", "06_random_search", "07_time_series_split", "08_production_pipeline"]
    ),
    "L33_Perceptron": (
        "Lesson 33: Perceptron",
        ["Understand biological inspiration", "Build single perceptron",
         "Recognize linear limitations", "Prepare for deep learning"],
        ["01_biological_neuron", "02_perceptron_model", "03_activation_threshold", "04_linear_boundary",
         "05_xor_problem", "06_perceptron_learning", "07_convergence", "08_finance_perceptron"]
    ),
    "L34_MLP_Activations": (
        "Lesson 34: MLPs and Activations",
        ["Design MLP architectures", "Choose activation functions",
         "Build models with Keras", "Apply to non-linear problems"],
        ["01_mlp_architecture", "02_relu_activation", "03_sigmoid_softmax", "04_keras_sequential",
         "05_hidden_layers", "06_parameter_counting", "07_universal_approximation", "08_market_regimes"]
    ),
    "L35_Backpropagation": (
        "Lesson 35: Backpropagation",
        ["Understand gradient descent", "Interpret loss curves",
         "Configure learning rate", "Monitor training progress"],
        ["01_forward_pass", "02_gradient_descent", "03_backprop_intuition", "04_loss_functions",
         "05_learning_rate", "06_training_curves", "07_batch_sizes", "08_volatility_prediction"]
    ),
    "L36_Overfitting_Prevention": (
        "Lesson 36: Overfitting Prevention",
        ["Apply dropout regularization", "Use early stopping",
         "Diagnose overfitting", "Build robust models"],
        ["01_overfitting_visual", "02_dropout_concept", "03_keras_dropout", "04_early_stopping",
         "05_validation_curves", "06_regularization_l2", "07_data_augmentation", "08_finance_regularization"]
    ),

    # Agent 7: L37-L42 (NLP + Deployment)
    "L37_Text_Preprocessing": (
        "Lesson 37: Text Preprocessing",
        ["Tokenize text", "Remove stopwords",
         "Apply stemming/lemmatization", "Clean financial text"],
        ["01_tokenization", "02_stopwords", "03_stemming_lemmatization", "04_nltk_basics",
         "05_regex_cleaning", "06_financial_text", "07_preprocessing_pipeline", "08_news_cleaning"]
    ),
    "L38_BOW_TFIDF": (
        "Lesson 38: BoW and TF-IDF",
        ["Create bag of words", "Calculate TF-IDF weights",
         "Build document vectors", "Apply to text classification"],
        ["01_bow_concept", "02_count_vectorizer", "03_tfidf_formula", "04_tfidf_vectorizer",
         "05_sparse_matrices", "06_vocabulary_size", "07_ngrams", "08_earnings_call_analysis"]
    ),
    "L39_Word_Embeddings": (
        "Lesson 39: Word Embeddings",
        ["Understand word vectors", "Use pre-trained Word2Vec",
         "Find similar words", "Create document embeddings"],
        ["01_embedding_concept", "02_word2vec", "03_gensim_usage", "04_similarity_search",
         "05_analogies", "06_document_vectors", "07_pretrained_models", "08_finance_embeddings"]
    ),
    "L40_Sentiment_Analysis": (
        "Lesson 40: Sentiment Analysis",
        ["Apply VADER sentiment", "Use FinBERT for finance",
         "Build sentiment classifiers", "Analyze market sentiment"],
        ["01_sentiment_concept", "02_vader_sentiment", "03_finbert_intro", "04_sentiment_classification",
         "05_aspect_sentiment", "06_time_series_sentiment", "07_news_sentiment", "08_trading_signals"]
    ),
    "L41_Model_Serialization": (
        "Lesson 41: Model Serialization",
        ["Save models with joblib", "Load and predict",
         "Version models", "Prepare for deployment"],
        ["01_serialization_concept", "02_pickle_joblib", "03_save_load_workflow", "04_model_versioning",
         "05_metadata_tracking", "06_sklearn_persistence", "07_keras_saving", "08_production_models"]
    ),
    "L42_FastAPI": (
        "Lesson 42: REST APIs with FastAPI",
        ["Create API endpoints", "Design input schemas",
         "Handle predictions", "Document with Swagger"],
        ["01_api_concept", "02_fastapi_basics", "03_endpoint_design", "04_pydantic_schemas",
         "05_prediction_endpoint", "06_error_handling", "07_swagger_docs", "08_finance_api"]
    ),

    # Agent 8: L43-L48 (Deployment + Project)
    "L43_Streamlit_Dashboards": (
        "Lesson 43: Streamlit Dashboards",
        ["Build interactive apps", "Add widgets and inputs",
         "Display predictions", "Create financial dashboards"],
        ["01_streamlit_intro", "02_widgets", "03_layouts", "04_caching",
         "05_charts_integration", "06_model_integration", "07_deployment_prep", "08_stock_dashboard"]
    ),
    "L44_Cloud_Deployment": (
        "Lesson 44: Cloud Deployment",
        ["Deploy to Streamlit Cloud", "Configure requirements",
         "Manage secrets", "Monitor applications"],
        ["01_cloud_options", "02_streamlit_cloud", "03_requirements_txt", "04_secrets_management",
         "05_github_integration", "06_deployment_workflow", "07_monitoring", "08_live_deployment"]
    ),
    "L45_Project_Work_1": (
        "Lesson 45: Project Work Session 1",
        ["Review project requirements", "Plan implementation",
         "Start coding", "Get feedback"],
        ["01_project_overview", "02_rubric_review", "03_topic_selection", "04_data_requirements",
         "05_model_planning", "06_deployment_planning", "07_timeline", "08_checkpoint_1"]
    ),
    "L46_Project_Work_2": (
        "Lesson 46: Project Work Session 2",
        ["Continue implementation", "Prepare presentation",
         "Practice demo", "Finalize code"],
        ["01_implementation_progress", "02_presentation_structure", "03_slide_design", "04_demo_planning",
         "05_code_cleanup", "06_documentation", "07_peer_feedback", "08_checkpoint_2"]
    ),
    "L47_ML_Ethics": (
        "Lesson 47: ML Ethics in Finance",
        ["Identify bias in models", "Ensure explainability",
         "Consider fairness", "Follow regulations"],
        ["01_ethics_importance", "02_bias_sources", "03_explainability", "04_shap_values",
         "05_fairness_metrics", "06_regulatory_requirements", "07_responsible_ai", "08_finance_ethics"]
    ),
    "L48_Final_Presentations": (
        "Lesson 48: Final Presentations",
        ["Present projects", "Demonstrate models",
         "Answer questions", "Receive feedback"],
        ["01_presentation_guidelines", "02_evaluation_criteria", "03_time_management", "04_demo_tips",
         "05_qa_handling", "06_peer_evaluation", "07_course_summary", "08_next_steps"]
    )
}

def generate_all():
    print("=" * 60)
    print("GENERATING L19-L48 (30 lessons, 240 charts)")
    print("=" * 60)

    for folder, (title, objectives, charts) in ALL_LESSONS.items():
        print(f"\n{folder}...")

        # Generate charts
        for i, chart in enumerate(charts, 1):
            create_chart(folder, chart)
            print(f"  Chart {i}/8: {chart}")

        # Generate .tex file
        tex_content = generate_tex(folder, title, objectives, charts)
        tex_path = BASE_DIR / folder / f"{folder}.tex"
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(tex_content)
        print(f"  Created: {tex_path}")

    print("\n" + "=" * 60)
    print("ALL L19-L48 COMPLETE (30 lessons, 240 charts)")
    print("=" * 60)

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

def generate_tex(folder, title, objectives, charts):
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
\bottomnote{Building towards your final project}
\end{frame}
""" % (title, "\n".join([f"\\item {obj}" for obj in objectives])))

    for chart in charts:
        chart_title = chart.replace('_', ' ').title()
        for i in range(1, 9):
            chart_title = chart_title.replace(f'{i:02d} ', '')
        slides.append(r"""
\begin{frame}[t]{%s}
\begin{center}
\includegraphics[width=0.85\textwidth]{%s/chart.pdf}
\end{center}
\end{frame}
""" % (chart_title, chart))

    slides.append(r"""
\begin{frame}[t]{Lesson Summary}
\textbf{Key Takeaways:}
\begin{itemize}
%s
\end{itemize}
\bottomnote{Apply these skills in your final project}
\end{frame}

\end{document}
""" % "\n".join([f"\\item {obj}" for obj in objectives]))

    return PREAMBLE + "\n".join(slides)

if __name__ == '__main__':
    generate_all()
