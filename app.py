import joblib
import streamlit as st

st.set_page_config(
    page_title="Drug Side-Effect Rating Classifier",
    page_icon="💊",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("side_effect_model.joblib")

st.title("💊 Drug Side-Effect Rating Classifier")
st.write(
    "An educational machine learning project that classifies "
    "patient-reported side-effect ratings from review text."
)

st.warning(
    "This prototype classifies historical review patterns. "
    "It does not diagnose conditions or predict an individual's "
    "medical risk. Do not use it to make treatment decisions."
)

model = load_model()

review = st.text_area(
    "Enter a patient review describing side effects:",
    placeholder=(
        "Example: I experienced nausea and dizziness "
        "after taking the medicine."
    ),
    height=150
)

if st.button("Classify review", type="primary"):
    if not review.strip():
        st.error("Please enter a review first.")
    else:
        prediction = model.predict([review])[0]
        st.subheader("Predicted reported rating")
        st.success(str(prediction))
        
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([review])[0]
            classes = model.classes_
            st.caption(
                "Model score (not a clinical probability):"
            )
            for label, score in sorted(
                zip(classes, probabilities),
                key=lambda item: item[1],
                reverse=True
            ):
                st.write(f"{label}: {score:.1%}")
                st.progress(float(score))

st.divider()
st.caption(
    "BTech Machine Learning Lab | Python, Scikit-learn and TF-IDF"
)
