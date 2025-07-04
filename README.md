**SOUND_CLUSTERING**



**Unlabeled Sound Data Clustering**


This repository contains the solution for a formative assignment focused on applying clustering techniques to an unlabeled sound dataset and exploring the application of Hidden Markov Models (HMMs) to a capstone project idea.

**Table of Contents**

*Part 1: Sound Dataset Clustering*

Objective

Dataset

Key Steps

Summary of Findings

Technologies Used

Part 2: Hidden Markov Model (HMM) Application

Capstone Project Idea

HMM Integration

How to Run the Notebook

Author

**Part 1: Sound Dataset Clustering**
Objective

The primary objective of this part of the assignment is to apply various clustering techniques to an unlabeled sound dataset, analyze the necessity and impact of dimensionality reduction, and compare different clustering methods based on performance metrics and visual interpretability.

Dataset

The dataset used for this assignment is an unlabeled collection of sound recordings, provided as unlabelled_sounds.zip. 

Upon extraction, the audio files are located in the unlabelled_sounds_extracted/unlabelled_sounds/ directory.

Key Steps

The analysis in the google colab Notebook (sound_clustering_assignment.ipynb) follows these key steps:

Data Loading & Feature Extraction:

The zipped dataset is extracted into a local directory.

Mel Spectrogram features are extracted from each sound file using the librosa library, resulting in a high-dimensional feature set (3000 samples, 128 features).

Initial Visualization (Without Dimensionality Reduction):

Attempts were made to visualize the raw 128-dimensional features using 2D scatter plots and pair plots.

Finding: Direct visualization proved impractical and uninterpretable due to the "Curse of Dimensionality," highlighting the necessity for dimensionality reduction.

Dimensionality Reduction (PCA & t-SNE):

The extracted features were standardized using StandardScaler.

Both Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) were applied to reduce the data to 3 components.

3D visualizations of the PCA and t-SNE transformed data were generated.

Finding: t-SNE provided visually clearer cluster separability compared to PCA, indicating its effectiveness in preserving local structures in the data.

Clustering the Sound Data (K-Means & DBSCAN):

K-Means Optimization: The optimal number of clusters (K) for K-Means was determined using the Elbow Method (based on inertia) and Silhouette Score analysis. An optimal K of 3 was selected.

Algorithm Application: K-Means was applied with K=3. DBSCAN was also applied with initial parameters (eps=0.5, min_samples=5).

Visual Interpretation: 2D scatter plots of the clusters (using t-SNE components for visualization) were generated for both K-Means and DBSCAN.

Evaluating Clustering Performance:

Quantitative metrics were used: Silhouette Score and Davies-Bouldin Index.

K-Means Performance: Achieved a Silhouette Score of 0.1684 and a Davies-Bouldin Index of 1.7438. While it successfully partitioned the data into 3 visually distinct clusters, the scores suggest moderate compactness and separation.

DBSCAN Performance: With the chosen parameters, DBSCAN failed to form any clusters, classifying all 3000 data points as noise. This demonstrates DBSCAN's high sensitivity to parameter tuning for the specific density of the dataset.

Summary of Findings

K-Means effectively clustered the sound data into 3 groups, which were visually interpretable on the t-SNE reduced dimensions. DBSCAN, however, struggled significantly with its default parameters, failing to identify any clusters. Dimensionality reduction, particularly t-SNE, was crucial for making the high-dimensional sound features visually comprehensible and for revealing underlying cluster structures.

Technologies Used

Python 3

librosa (for audio feature extraction)

numpy (for numerical operations)

pandas (for data manipulation)

matplotlib (for plotting)

seaborn (for enhanced visualizations)

scikit-learn (for StandardScaler, PCA, TSNE, KMeans, DBSCAN, silhouette_score, davies_bouldin_score)

**Part 2: Hidden Markov Model (HMM) Application**

Capstone Project Idea
The capstone project aims to develop a low-bandwidth educational content platform using HTML, CSS, JavaScript, and Node.js. This "Web-Based Quiz and E-Learning Platform with Offline Support" will allow children in rural areas to study even with poor internet connection, aligning with SDG 4 (Quality Education).

HMM Integration

A Hidden Markov Model (HMM) could be employed in this platform to model and predict a student's underlying knowledge state (the hidden states) based on their observable interactions, thereby enabling adaptive learning and optimized content delivery (e.g., pre-caching).

Describe the Observations:

The measurable data (observations) for the HMM would be the observable actions and responses of students on the platform. These include:

Quiz Answers: Sequences of correct or incorrect responses to questions.

Time Taken: Duration spent on questions or learning modules.

Attempt Count: Number of attempts made on a specific question.

Navigation Patterns: The sequence of learning modules or topics accessed.

Type of HMM Problem:

Since the actual knowledge state of a student (e.g., "Novice," "Mastered") is not directly known or labeled in advance, this represents a Learning (or Training) Problem for the HMM. The model needs to learn the probabilistic relationships between observable actions and hidden knowledge states.

Training Algorithm:

The Baum-Welch algorithm, an Expectation-Maximization (EM) algorithm, is the standard choice for the HMM Learning Problem.

a. What values are known at the start?

Observation Sequences (O): The recorded sequences of student interactions (e.g., quiz answers).

Number of Hidden States (N): A pre-defined number of knowledge levels (e.g., 2 or 3 states like "Not Mastered," "Partially Mastered," "Mastered").

Number of Unique Observations (M): The size of the observation alphabet (e.g., "Correct", "Incorrect").

Initial Estimates: Randomly initialized values for the HMM parameters.

b. What values are unknown and need to be learned?

The Baum-Welch algorithm learns the core HMM parameters, often denoted as 
lambda=(A,B,
pi):

State Transition Probability Matrix (A): The probabilities of a student transitioning from one hidden knowledge state to another.

Observation (Emission) Probability Matrix (B): The probabilities of observing a specific student action (e.g., a correct answer) given a particular hidden knowledge state.

Initial State Probabilities (
pi): The probabilities of a student starting in each hidden knowledge state.

c. Parameter Updates:

The Baum-Welch algorithm iteratively updates all three types of HMM parameters (
pi, A, and B) to maximize the likelihood of the observed sequences, converging to a local optimum.

How to Run the Notebook

Clone the Repository:

git clone https://github.com/IkireziI/Sound_Clustering.git

cd Sound_Clustering

Download Dataset:

Download unlabelled_sounds.zip

from my repository in the folder called Datasets.

Install Dependencies:

pip install numpy pandas librosa scikit-learn matplotlib seaborn

Open google colab Notebook:

google colab notebook sound_clustering_assignment.ipynb

Run Cells:

Execute all cells in the notebook sequentially. 

The notebook includes all necessary code for data loading, feature extraction, dimensionality reduction, clustering, and evaluation.





Author


Inès IKIREZI


IkireziI
