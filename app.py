from flask import Flask, render_template, request
import io
import threading
import webbrowser
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


def load_dataset(file_storage, expected_suffix):
    filename = file_storage.filename or ""
    if not filename.lower().endswith(expected_suffix):
        raise ValueError(f"Please choose a {expected_suffix} file for this upload.")
    try:
        frame = pd.read_csv(io.BytesIO(file_storage.read()), header=None)
    except Exception as exc:
        raise ValueError(f"Could not read {filename}. Check that it is a comma-separated data file.") from exc
    if frame.shape[1] != 65:
        raise ValueError(f"{filename} has {frame.shape[1]} columns; expected 65 (64 pixel values and one label).")
    if frame.empty or frame.isna().any().any():
        raise ValueError(f"{filename} is empty or contains missing values.")
    try:
        values = frame.apply(pd.to_numeric, errors="raise")
    except Exception as exc:
        raise ValueError(f"{filename} contains non-numeric values. Check its contents and delimiter.") from exc
    return values.iloc[:, :64], values.iloc[:, 64]


@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    error = None
    if request.method == "POST":
        train_file = request.files.get("train_file")
        test_file = request.files.get("test_file")
        if not train_file or not train_file.filename or not test_file or not test_file.filename:
            error = "Select both the training .tra file and the test .tes file."
        else:
            try:
                x_train, y_train = load_dataset(train_file, ".tra")
                x_test, y_test = load_dataset(test_file, ".tes")
                model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale"))
                model.fit(x_train, y_train)
                predictions = model.predict(x_test)
                labels = list(range(10))
                report = classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0)
                matrix = confusion_matrix(y_test, predictions, labels=labels).tolist()
                results = {
                    "train_count": len(x_train), "test_count": len(x_test),
                    "accuracy": accuracy_score(y_test, predictions),
                    "digits": [{"digit": str(d), "precision": report[str(d)]["precision"],
                                "recall": report[str(d)]["recall"], "f1": report[str(d)]["f1-score"],
                                "support": int(report[str(d)]["support"])} for d in labels],
                    "matrix": matrix,
                }
            except ValueError as exc:
                error = str(exc)
            except Exception as exc:
                error = f"Model could not run: {exc}"
    return render_template("index.html", results=results, error=error)


@app.errorhandler(413)
def too_large(_error):
    return render_template("index.html", results=None, error="The selected files exceed the 32 MB upload limit."), 413


if __name__ == "__main__":
    url = "http://127.0.0.1:5000"
    threading.Timer(1.5, lambda: webbrowser.open_new_tab(url)).start()
    print(f"DigitLab is starting at {url}. Leave this window open.", flush=True)
    app.run(host="127.0.0.1", port=5000, debug=False)



