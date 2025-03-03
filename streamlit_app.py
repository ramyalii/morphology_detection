

import streamlit as st
from PIL import Image
import time
import io

# Sidebar: About section
st.sidebar.title("About")
st.sidebar.info("""
This application uses a computer vision model to classify microscopic images of microalgae (Spirulina).
It helps researchers and R&D teams assess the quality of microalgae strains.
""")

# Main title and description
st.title("🌱 Microalgae Morphology Detection")
st.markdown("""
Welcome to the **Microalgae Morphology Detection** app! This tool uses a computer vision model to classify whether a given microscopic image is of **Spirulina** or not.
Upload an image to get started.
""")

# Confidence threshold slider
confidence_threshold = st.slider(
    "Set Confidence Threshold (%)",
    min_value=50,
    max_value=100,
    value=90,
    step=1
)

# Tabs for single and multiple image uploads
tab1, tab2 = st.tabs(["Single Image Upload", "Multiple Image Upload"])

# History for predictions
if "history" not in st.session_state:
    st.session_state["history"] = []

# Single Image Upload Section
with tab1:
    st.header("Upload a Single Microscopic Image")
    uploaded_file = st.file_uploader("Upload a Microscopic Image", type=["jpg", "jpeg", "png"], key="single")

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Microscopic Image", use_column_width=True)
        st.write("Processing the image...")

        # Simulate a progress bar
        with st.spinner("Analyzing the image..."):
            time.sleep(2)  # Simulate processing time

        # Placeholder for model prediction (to be connected to the backend)
        # Simulating a response for demonstration purposes
        prediction = "Spirulina"  # Example: Replace with model's output
        confidence = 0.95  # Example: Replace with model's confidence score

        # Check against confidence threshold
        if confidence * 100 >= confidence_threshold:
            result = f"**Classification:** {prediction} (Confidence: {confidence * 100:.2f}%)"
        else:
            result = f"**Classification:** Not confident enough (Confidence: {confidence * 100:.2f}%)"

        # Display the result
        st.write("### Prediction Result:")
        st.write(result)

        # Add to history
        st.session_state["history"].append({
            "image": uploaded_file.name,
            "prediction": prediction,
            "confidence": confidence * 100
        })

        # Generate a downloadable report
        report = f"""
        Microalgae Morphology Detection Report
        --------------------------------------
        Image: {uploaded_file.name}
        Classification: {prediction}
        Confidence: {confidence * 100:.2f}%
        """
        report_bytes = io.BytesIO(report.encode("utf-8"))
        st.download_button(
            label="Download Report",
            data=report_bytes,
            file_name="microalgae_report.txt",
            mime="text/plain"
        )

# Multiple Image Upload Section
with tab2:
    st.header("Upload Multiple Microscopic Images")
    uploaded_files = st.file_uploader(
        "Upload Microscopic Images (You can select multiple files)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="multiple"
    )

    if uploaded_files:
        results = []
        for uploaded_file in uploaded_files:
            # Display each uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

            # Simulate a progress bar for each image
            with st.spinner(f"Analyzing {uploaded_file.name}..."):
                time.sleep(2)  # Simulate processing time

            # Placeholder for model prediction (to be connected to the backend)
            # Simulating a response for demonstration purposes
            prediction = "Spirulina"  # Example: Replace with model's output
            confidence = 0.95  # Example: Replace with model's confidence score

            # Check against confidence threshold
            if confidence * 100 >= confidence_threshold:
                result = f"**Classification:** {prediction} (Confidence: {confidence * 100:.2f}%)"
            else:
                result = f"**Classification:** Not confident enough (Confidence: {confidence * 100:.2f}%)"

            # Display the result for each image
            st.write(f"### Result for {uploaded_file.name}:")
            st.write(result)

            # Add to history
            st.session_state["history"].append({
                "image": uploaded_file.name,
                "prediction": prediction,
                "confidence": confidence * 100
            })

            # Append to results for report generation
            results.append({
                "image": uploaded_file.name,
                "prediction": prediction,
                "confidence": confidence * 100
            })

        # Generate a downloadable report for all images
        if results:
            report = "Microalgae Morphology Detection Report\n"
            report += "--------------------------------------\n"
            for result in results:
                report += f"Image: {result['image']}\n"
                report += f"Classification: {result['prediction']}\n"
                report += f"Confidence: {result['confidence']:.2f}%\n"
                report += "--------------------------------------\n"
            report_bytes = io.BytesIO(report.encode("utf-8"))
            st.download_button(
                label="Download Report for All Images",
                data=report_bytes,
                file_name="microalgae_report_multiple.txt",
                mime="text/plain"
            )

# Display history
if st.session_state["history"]:
    st.write("### Prediction History:")
    for i, entry in enumerate(st.session_state["history"]):
        st.write(f"**{i+1}. Image:** {entry['image']} | **Prediction:** {entry['prediction']} | **Confidence:** {entry['confidence']:.2f}%")

