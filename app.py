from pathlib import Path
import zipfile

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "starbucks_rating_model.joblib"
DATA_PATH = BASE_DIR / "archive.zip"

st.set_page_config(page_title="Starbucks Rating Predictor", page_icon="☕")
st.title("Starbucks Rating Predictor")
st.write("Enter a Starbucks review to predict its rating.")

@st.cache_resource
def load_resources():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Upload it beside app.py."
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATA_PATH.name}. "
            "Upload it beside app.py."
        )

    with zipfile.ZipFile(DATA_PATH, "r") as archive:
        csv_name = next(
            (name for name in archive.namelist()
             if Path(name).name == "reviews_data.csv"),
            None,
        )
        if csv_name is None:
            raise FileNotFoundError(
                "Could not find reviews_data.csv inside archive.zip."
            )
        with archive.open(csv_name) as csv_file:
            df = pd.read_csv(csv_file)

    if "Review" not in df.columns or "Rating" not in df.columns:
        raise ValueError(
            "The dataset must contain 'Review' and 'Rating' columns."
        )

    clean_df = df.dropna(subset=["Review", "Rating"]).copy()
    clean_df["Review"] = clean_df["Review"].astype(str)
    clean_df["Rating"] = clean_df["Rating"].astype(int)

    # Recreate the same TF-IDF features used when training the model.
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    vectorizer.fit_transform(clean_df["Review"])

    model = joblib.load(MODEL_PATH)
    return model, vectorizer

try:
    model, vectorizer = load_resources()
except Exception as error:
    st.error(f"Could not load the model or dataset: {error}")
    st.stop()

review = st.text_area(
    "Enter review",
    placeholder="Example: The coffee was great and the staff were friendly.",
)

if st.button("Predict rating"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        features = vectorizer.transform([review])
        prediction = model.predict(features)
        st.success(f"Predicted rating: {int(prediction[0])} star(s)")
