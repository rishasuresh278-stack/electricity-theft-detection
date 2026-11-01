
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
import tempfile

# Load trained models
bundle = joblib.load("electricity_theft_models.joblib")

improved_model = bundle["improved_model"]
iso_model = bundle["iso_model"]
imputer = bundle["imputer"]
feature_names = bundle["feature_names"]
date_columns = bundle["date_columns"]

rf_threshold = float(bundle["best_threshold"])
iso_threshold = float(bundle["best_iso_threshold"])


def extract_features(data):

    available_dates = [
        col for col in date_columns if col in data.columns
    ]

    if not available_dates:
        raise ValueError(
            "No matching daily consumption columns found in the CSV."
        )

    consumption = data[available_dates].apply(
        pd.to_numeric, errors="coerce"
    )

    features = pd.DataFrame(index=data.index)

    features["mean_consumption"] = consumption.mean(axis=1)
    features["std_consumption"] = consumption.std(axis=1)
    features["min_consumption"] = consumption.min(axis=1)
    features["max_consumption"] = consumption.max(axis=1)
    features["median_consumption"] = consumption.median(axis=1)
    features["zero_days"] = (consumption == 0).sum(axis=1)
    features["missing_days"] = consumption.isna().sum(axis=1)
    features["total_consumption"] = consumption.sum(axis=1)

    mean_values = features["mean_consumption"].replace(0, np.nan)

    features["consumption_cv"] = (
        features["std_consumption"] / mean_values
    )

    return features[feature_names]


def detect_theft(file, model_choice):

    if file is None:
        raise gr.Error("Please upload an electricity CSV file.")

    try:
        data = pd.read_csv(file)

        if data.empty:
            raise ValueError("The uploaded CSV is empty.")

        features = extract_features(data)

        X = imputer.transform(features)
        X = pd.DataFrame(X, columns=feature_names)

        results = data.copy()

        if model_choice == "Random Forest":

            probabilities = improved_model.predict_proba(X)[:, 1]

            predictions = (
                probabilities >= rf_threshold
            ).astype(int)

            results["Theft_Risk_Score"] = np.round(
                probabilities * 100, 2
            )

        else:

            anomaly_scores = -iso_model.decision_function(X)

            predictions = (
                anomaly_scores >= iso_threshold
            ).astype(int)

            results["Anomaly_Score"] = np.round(
                anomaly_scores, 5
            )

        results["Prediction"] = np.where(
            predictions == 1,
            "Suspicious",
            "Normal"
        )

        suspicious_count = int(predictions.sum())
        normal_count = len(predictions) - suspicious_count

        summary = f"""
        ELECTRICITY THEFT DETECTION REPORT

        Total consumers analysed: {len(results)}

        Normal consumers: {normal_count}

        Suspicious consumers: {suspicious_count}

        Model used: {model_choice}
        """

        suspicious_results = results[
            results["Prediction"] == "Suspicious"
        ]

        output_file = os.path.join(
            tempfile.gettempdir(),
            "electricity_theft_report.csv"
        )

        results.to_csv(output_file, index=False)

        return (
            summary,
            results.head(100),
            suspicious_results.head(100),
            output_file
        )

    except Exception as e:
        raise gr.Error(f"Error processing CSV: {str(e)}")


with gr.Blocks(title="Electricity Theft Detection") as demo:

    gr.Markdown(
        """
        # ⚡ Electricity Theft Detection System

        ### Machine Learning Based Electricity Consumption Analysis

        Upload an electricity consumption CSV file to identify
        potentially suspicious consumers.
        """
    )

    file_input = gr.File(
        label="Upload Electricity CSV",
        file_types=[".csv"],
        type="filepath"
    )

    model_choice = gr.Radio(
        choices=["Random Forest", "Isolation Forest"],
        value="Random Forest",
        label="Choose Machine Learning Model"
    )

    analyze_button = gr.Button(
        "Analyze Electricity Data",
        variant="primary"
    )

    summary_output = gr.Textbox(
        label="Analysis Summary",
        lines=8
    )

    results_output = gr.Dataframe(
        label="Consumer Prediction Results"
    )

    suspicious_output = gr.Dataframe(
        label="Suspicious Consumers"
    )

    download_output = gr.File(
        label="Download Complete Prediction Report"
    )

    analyze_button.click(
        fn=detect_theft,
        inputs=[file_input, model_choice],
        outputs=[
            summary_output,
            results_output,
            suspicious_output,
            download_output
        ]
    )

    gr.Markdown(
        """
        **Disclaimer:** Predictions are machine-learning flags for
        further investigation, not proof of electricity theft.
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
