# Telco Customer Churn Prediction Machine Learning System

This repository provides an end-to-end Machine Learning solution to detect high-risk churn candidates in a telecom ecosystem.

##  Model Blueprint & Performance

| Pipeline Configuration | Value / Setting | Core Evaluation Metric | Production Result |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | XGBoost Classifier |  **ROC-AUC** | **80.31%** |
| **Hyperparameters** | `n_estimators=600`, `max_depth=4` |  **F1-Score (Churn)** | **59.0%** |
| **Imbalance Strategy** | `scale_pos_weight` (2.77) | **Recall (Churn)** | **64.0%** |
| **Data Split** | 80/20 Stratified Split | **Accuracy Baseline** | 77.0% |

## Setup Instructions

1. **Clone the repository:**
   
   git clone [https://github.com/slhhnin/telco-churn-prediction-.git](https://github.com/slhhnin/telco-churn-prediction-.git)
   
   cd telco-churn-prediction

2.  **Create and Activate a Virtual Environment:**

    **On macOS/Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

    **On Windows (Command Prompt/PowerShell):**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Explore the Data:**

    Open ``analysis.ipynb`` in your Jupyter environment to review EDA and data distribution charts.

5. **Train the Model:**

    Run the pipeline to process data, counter class imbalances, and export model weights:
    ```bash
    python train.py
    ```
6. **Spin up the Prediction API:**

   1. **Launch the Server**

        Run the following command from the root directory of the project:
        ```bash
        uvicorn app:app --reload
        ```
        Access interactive API docs at: http://127.0.0.1:8000/docs

    2. **Test the API Endpoints**

        You can verify that the system handles batch processing correctly using either your web browser or your system terminal.

        **Method A: Interactive Web UI (Swagger Docs)**
        - Open your web browser and navigate to: http://127.0.0.1:8000/docs
        - Click on the ``POST /predict`` dropdown.
        - Click ``Try it out``.
        - Open the ``sample_request.json`` file from this repository, copy its entire contents (the array of 10 items), paste it into the request body interface box, and click Execute.

        **Method B: Terminal Testing via cURL**

        - Open a separate terminal window and stream the ``sample_request.json`` directly into the prediction layer:

        - On macOS/Linux:
            ```Bash
            curl -X 'POST' 'http://127.0.0.1:8000/predict' -H 'accept: application/json' -H 'Content-Type: application/json' -d @sample_request.json
            ```

            SSH