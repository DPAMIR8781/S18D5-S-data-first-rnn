Employee Salary Forecasting with RNN & LSTM

This project focuses on time-series forecasting using Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks to predict employee salaries based on historical career progression data.

The project combines deep learning with exploratory workforce analytics to analyze salary growth trajectories, employee segmentation, and temporal compensation patterns.

Project Overview

The dataset represents synthetic employee career histories over 10 years.

Each employee sequence contains:

Salary progression
Number of employees under responsibility
Company size

The objective is to predict the employee’s salary in the 11th year using previous yearly observations.

Dataset Shape:

X.shape = (25000, 10, 3)
y.shape = (25000,)

Meaning:

25,000 employee sequences
10 yearly time steps
3 features per year
Technologies Used
Python
NumPy
Pandas
Matplotlib
Seaborn
TensorFlow / Keras
Scikit-learn
Deep Learning Models
SimpleRNN Model

The first model was built using:

Normalization layer
SimpleRNN layer with tanh activation
Dense hidden layer
Regression output layer
Model Architecture
model = Sequential([
    Input(shape=(10, 3)),
    normalizer,
    SimpleRNN(20, activation='tanh'),
    Dense(10, activation='relu'),
    Dense(1)
])
Compilation
model.compile(
    optimizer='rmsprop',
    loss='mse',
    metrics=['mae']
)
LSTM Model

An additional LSTM architecture was implemented to compare sequence-learning performance against the SimpleRNN model.

lstm_model = Sequential([
    Input(shape=(10, 3)),
    normalizer_lstm,
    LSTM(20, activation='tanh'),
    Dense(10, activation='relu'),
    Dense(1)
])
Training Strategy
Validation Split: 20%
EarlyStopping callback
Restore best weights enabled
Batch Size: 32
Up to 100 epochs
EarlyStopping(
    patience=5,
    restore_best_weights=True
)
Model Evaluation
Baseline Comparison

A baseline model was created assuming:

“The 11th year salary will remain equal to the 10th year salary.”

Baseline MAE
Baseline MAE: 0.588
RNN MAE
RNN MAE: 0.278

The RNN model significantly outperformed the baseline prediction approach.

Exploratory Data Analysis (EDA)

Beyond model training, additional exploratory workforce analytics were performed.

Average Salary Growth Across Employees

The average employee salary steadily increases over time, demonstrating realistic temporal salary progression patterns.

Insights:

Salaries consistently grow across years
Growth accelerates after early career stages
Later years show more stable linear progression
Salary Growth Distribution

Salary growth rates were analyzed across all employees.

Key observations:

Most employees exhibit low-to-medium salary growth
A smaller group demonstrates extremely high salary acceleration
The distribution is heavily right-skewed
Extreme outliers represent rapid career progression scenarios

A logarithmic scale was also used to better visualize outlier behavior.

Employee Salary Growth Segmentation

Employees were segmented into growth categories:

Low Growth (<100%)
Medium Growth (100–500%)
High Growth (>500%)

This segmentation demonstrates workforce diversity and varying career progression trajectories.

Top Salary Growth Employees

The project identifies employees with the highest salary acceleration over the 10-year period.

This analysis helps visualize:

rapid promotions,
leadership transitions,
and strong company growth effects.
Most Stable Salary Employees

Additional analysis was performed to identify employees with the lowest salary volatility.

These sequences demonstrate highly stable career paths and predictable salary progression patterns.

Time-Series Insights

The dataset contains realistic sequential patterns suitable for recurrent neural networks:

Non-linear salary progression
Career acceleration periods
Stable compensation trajectories
Long-term temporal dependencies

These properties make the dataset highly appropriate for RNN and LSTM architectures.

Key Learning Outcomes

This project helped strengthen understanding of:

Time-series data structures
Sequential deep learning models
RNN and LSTM architectures
Regression-based forecasting
EarlyStopping techniques
Baseline model comparison
Exploratory workforce analytics
Temporal salary trend analysis
Repository Structure
├── first_rnn.ipynb
├── X.npy
├── y.npy
├── README.md
Future Improvements

Possible future enhancements include:

GRU architectures
Attention mechanisms
Hyperparameter optimization
Prediction error analysis
Salary clustering
Promotion jump detection
Interactive dashboards
Transfer learning approaches
Author

Doruk Pamir

Aspiring Data Analyst & Data Scientist passionate about:

Machine Learning
Deep Learning
Time-Series Forecasting
Workforce Analytics
Data Visualization
Predictive Modeling
