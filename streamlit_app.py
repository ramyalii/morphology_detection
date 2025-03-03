
# import streamlit as st
# from PIL import Image

# # Title and description
# st.title("Microalgae Morphology Detection")
# st.write("""
# This application uses a computer vision classification model to detect whether the given microscopic image is of Spirulina or not.
# Upload a microscopic image to get started.
# """)

# # File uploader for image input
# uploaded_file = st.file_uploader("Upload a Microscopic Image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Microscopic Image", use_column_width=True)
#     st.write("Processing the image...")

#     # Placeholder for model prediction (to be connected to the backend)
#     # Simulating a response for demonstration purposes
#     # Replace this with actual model inference
#     prediction = "Spirulina"  # Example: Replace with model's output
#     confidence = 0.95  # Example: Replace with model's confidence score

#     # Display the result
#     st.write("### Prediction Result:")
#     st.write(f"**Classification:** {prediction}")
#     st.write(f"**Confidence:** {confidence * 100:.2f}%")
# else:
#     st.write("Please upload a microscopic image to proceed.")



# import streamlit as st
# from PIL import Image
# import time
# import io

# # Sidebar for navigation and information
# st.sidebar.title("Navigation")
# st.sidebar.markdown("""
# - **Home**: Upload and classify images.
# - **About**: Learn more about the project.
# - **Contact**: Reach out to us.
# """)

# # Sidebar: About section
# st.sidebar.title("About")
# st.sidebar.info("""
# This application uses a computer vision model to classify microscopic images of microalgae (Spirulina).
# It helps researchers and R&D teams assess the quality of microalgae strains.
# """)

# # Sidebar: Contact section
# st.sidebar.title("Contact")
# st.sidebar.info("""
# For inquiries, please contact:
# - **Email**: research@microalgae.com
# - **Phone**: +1-234-567-890
# """)

# # Main title and description
# st.title("🌱 Microalgae Morphology Detection")
# st.markdown("""
# Welcome to the **Microalgae Morphology Detection** app! This tool uses a computer vision model to classify whether a given microscopic image is of **Spirulina** or not.
# Upload an image to get started.
# """)

# # File uploader for image input
# uploaded_file = st.file_uploader("Upload a Microscopic Image", type=["jpg", "jpeg", "png"])

# # Confidence threshold slider
# confidence_threshold = st.slider(
#     "Set Confidence Threshold (%)",
#     min_value=50,
#     max_value=100,
#     value=90,
#     step=1
# )

# # History section
# if "history" not in st.session_state:
#     st.session_state["history"] = []

# if uploaded_file is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Microscopic Image", use_column_width=True)
#     st.write("Processing the image...")

#     # Simulate a progress bar
#     with st.spinner("Analyzing the image..."):
#         time.sleep(2)  # Simulate processing time

#     # Placeholder for model prediction (to be connected to the backend)
#     # Simulating a response for demonstration purposes
#     prediction = "Spirulina"  # Example: Replace with model's output
#     confidence = 0.95  # Example: Replace with model's confidence score

#     # Check against confidence threshold
#     if confidence * 100 >= confidence_threshold:
#         result = f"**Classification:** {prediction} (Confidence: {confidence * 100:.2f}%)"
#     else:
#         result = f"**Classification:** Not confident enough (Confidence: {confidence * 100:.2f}%)"

#     # Display the result
#     st.write("### Prediction Result:")
#     st.write(result)

#     # Add to history
#     st.session_state["history"].append({
#         "image": uploaded_file.name,
#         "prediction": prediction,
#         "confidence": confidence * 100
#     })

#     # Generate a downloadable report
#     report = f"""
#     Microalgae Morphology Detection Report
#     --------------------------------------
#     Image: {uploaded_file.name}
#     Classification: {prediction}
#     Confidence: {confidence * 100:.2f}%
#     """
#     report_bytes = io.BytesIO(report.encode("utf-8"))
#     st.download_button(
#         label="Download Report",
#         data=report_bytes,
#         file_name="microalgae_report.txt",
#         mime="text/plain"
#     )
# else:
#     st.write("Please upload a microscopic image to proceed.")

# # Display history
# if st.session_state["history"]:
#     st.write("### Prediction History:")
#     for i, entry in enumerate(st.session_state["history"]):
#         st.write(f"**{i+1}. Image:** {entry['image']} | **Prediction:** {entry['prediction']} | **Confidence:** {entry['confidence']:.2f}%")


# import streamlit as st
# from PIL import Image
# import time
# import io

# # Sidebar for navigation and information
# st.sidebar.title("Navigation")
# st.sidebar.markdown("""
# - **Home**: Upload and classify images.
# - **About**: Learn more about the project.
# - **Contact**: Reach out to us.
# """)

# # Sidebar: About section
# st.sidebar.title("About")
# st.sidebar.info("""
# This application uses a computer vision model to classify microscopic images of microalgae (Spirulina).
# It helps researchers and R&D teams assess the quality of microalgae strains.
# """)

# # Sidebar: Contact section
# st.sidebar.title("Contact")
# st.sidebar.info("""
# For inquiries, please contact:
# - **Email**: research@microalgae.com
# - **Phone**: +1-234-567-890
# """)

# # Main title and description
# st.title("🌱 Microalgae Morphology Detection")
# st.markdown("""
# Welcome to the **Microalgae Morphology Detection** app! This tool uses a computer vision model to classify whether a given microscopic image is of **Spirulina** or not.
# Upload an image to get started.
# """)

# # Confidence threshold slider
# confidence_threshold = st.slider(
#     "Set Confidence Threshold (%)",
#     min_value=50,
#     max_value=100,
#     value=90,
#     step=1
# )

# # Tabs for single and multiple image uploads
# tab1, tab2 = st.tabs(["Single Image Upload", "Multiple Image Upload"])

# # History for predictions
# if "history" not in st.session_state:
#     st.session_state["history"] = []

# # Single Image Upload Section
# with tab1:
#     st.header("Upload a Single Microscopic Image")
#     uploaded_file = st.file_uploader("Upload a Microscopic Image", type=["jpg", "jpeg", "png"], key="single")

#     if uploaded_file is not None:
#         # Display the uploaded image
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Microscopic Image", use_column_width=True)
#         st.write("Processing the image...")

#         # Simulate a progress bar
#         with st.spinner("Analyzing the image..."):
#             time.sleep(2)  # Simulate processing time

#         # Placeholder for model prediction (to be connected to the backend)
#         # Simulating a response for demonstration purposes
#         prediction = "Spirulina"  # Example: Replace with model's output
#         confidence = 0.95  # Example: Replace with model's confidence score

#         # Check against confidence threshold
#         if confidence * 100 >= confidence_threshold:
#             result = f"**Classification:** {prediction} (Confidence: {confidence * 100:.2f}%)"
#         else:
#             result = f"**Classification:** Not confident enough (Confidence: {confidence * 100:.2f}%)"

#         # Display the result
#         st.write("### Prediction Result:")
#         st.write(result)

#         # Add to history
#         st.session_state["history"].append({
#             "image": uploaded_file.name,
#             "prediction": prediction,
#             "confidence": confidence * 100
#         })

#         # Generate a downloadable report
#         report = f"""
#         Microalgae Morphology Detection Report
#         --------------------------------------
#         Image: {uploaded_file.name}
#         Classification: {prediction}
#         Confidence: {confidence * 100:.2f}%
#         """
#         report_bytes = io.BytesIO(report.encode("utf-8"))
#         st.download_button(
#             label="Download Report",
#             data=report_bytes,
#             file_name="microalgae_report.txt",
#             mime="text/plain"
#         )

# # Multiple Image Upload Section
# with tab2:
#     st.header("Upload Multiple Microscopic Images")
#     uploaded_files = st.file_uploader(
#         "Upload Microscopic Images (You can select multiple files)",
#         type=["jpg", "jpeg", "png"],
#         accept_multiple_files=True,
#         key="multiple"
#     )

#     if uploaded_files:
#         results = []
#         for uploaded_file in uploaded_files:
#             # Display each uploaded image
#             image = Image.open(uploaded_file)
#             st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

#             # Simulate a progress bar for each image
#             with st.spinner(f"Analyzing {uploaded_file.name}..."):
#                 time.sleep(2)  # Simulate processing time

#             # Placeholder for model prediction (to be connected to the backend)
#             # Simulating a response for demonstration purposes
#             prediction = "Spirulina"  # Example: Replace with model's output
#             confidence = 0.95  # Example: Replace with model's confidence score

#             # Check against confidence threshold
#             if confidence * 100 >= confidence_threshold:
#                 result = f"**Classification:** {prediction} (Confidence: {confidence * 100:.2f}%)"
#             else:
#                 result = f"**Classification:** Not confident enough (Confidence: {confidence * 100:.2f}%)"

#             # Display the result for each image
#             st.write(f"### Result for {uploaded_file.name}:")
#             st.write(result)

#             # Add to history
#             st.session_state["history"].append({
#                 "image": uploaded_file.name,
#                 "prediction": prediction,
#                 "confidence": confidence * 100
#             })

#             # Append to results for report generation
#             results.append({
#                 "image": uploaded_file.name,
#                 "prediction": prediction,
#                 "confidence": confidence * 100
#             })

#         # Generate a downloadable report for all images
#         if results:
#             report = "Microalgae Morphology Detection Report\n"
#             report += "--------------------------------------\n"
#             for result in results:
#                 report += f"Image: {result['image']}\n"
#                 report += f"Classification: {result['prediction']}\n"
#                 report += f"Confidence: {result['confidence']:.2f}%\n"
#                 report += "--------------------------------------\n"
#             report_bytes = io.BytesIO(report.encode("utf-8"))
#             st.download_button(
#                 label="Download Report for All Images",
#                 data=report_bytes,
#                 file_name="microalgae_report_multiple.txt",
#                 mime="text/plain"
#             )

# # Display history
# if st.session_state["history"]:
#     st.write("### Prediction History:")
#     for i, entry in enumerate(st.session_state["history"]):
#         st.write(f"**{i+1}. Image:** {entry['image']} | **Prediction:** {entry['prediction']} | **Confidence:** {entry['confidence']:.2f}%")



# import streamlit as st
# from PIL import Image
# import time
# import io

# # Sidebar: About section
# st.sidebar.title("About")
# st.sidebar.info("""
# This application uses a computer vision model to classify microscopic images of microalgae (Spirulina).
# It helps researchers and R&D teams assess the quality of microalgae strains.
# """)

# # Main title and description
# st.title("🌱 Microalgae Morphology Detection")
# st.markdown("""
# Welcome to the **Microalgae Morphology Detection** app! This tool uses a computer vision model to classify whether a given microscopic image is of **Spirulina** or not.
# Upload an image to get started.
# """)

# # Confidence threshold slider
# confidence_threshold = st.slider(
#     "Set Confidence Threshold (%)",
#     min_value=50,
#     max_value=100,
#     value=90,
#     step=1
# )

# # Tabs for single and multiple image uploads
# tab1, tab2 = st.tabs(["Single Image Upload", "Multiple Image Upload"])

# # History for predictions
# if "history" not in st.session_state:
#     st.session_state["history"] = []

# # Single Image Upload Section
# with tab1:
#     st.header("Upload a Single Microscopic Image")
#     uploaded_file = st.file_uploader("Upload a Microscopic Image", type=["jpg", "jpeg", "png"], key="single")

#     if uploaded_file is not None:
#         # Display the uploaded image
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Microscopic Image", use_column_width=True)
#         st.write("Processing the image...")

#         # Simulate a progress bar
#         with st.spinner("Analyzing the image..."):
#             time.sleep(2)  # Simulate processing time

#         # Placeholder for model prediction (to be connected to the backend)
#         # Simulating a response for demonstration purposes
#         prediction = "Spirulina"  # Example: Replace with model's output
#         confidence = 0.95  # Example: Replace with model's confidence score

#         # Check against confidence threshold
#         if confidence * 100 >= confidence_threshold:
#             result = f"**Classification:** {prediction} (Confidence: {confidence * 100:.2f}%)"
#         else:
#             result = f"**Classification:** Not confident enough (Confidence: {confidence * 100:.2f}%)"

#         # Display the result
#         st.write("### Prediction Result:")
#         st.write(result)

#         # Add to history
#         st.session_state["history"].append({
#             "image": uploaded_file.name,
#             "prediction": prediction,
#             "confidence": confidence * 100
#         })

#         # Generate a downloadable report
#         report = f"""
#         Microalgae Morphology Detection Report
#         --------------------------------------
#         Image: {uploaded_file.name}
#         Classification: {prediction}
#         Confidence: {confidence * 100:.2f}%
#         """
#         report_bytes = io.BytesIO(report.encode("utf-8"))
#         st.download_button(
#             label="Download Report",
#             data=report_bytes,
#             file_name="microalgae_report.txt",
#             mime="text/plain"
#         )

# # Multiple Image Upload Section
# with tab2:
#     st.header("Upload Multiple Microscopic Images")
#     uploaded_files = st.file_uploader(
#         "Upload Microscopic Images (You can select multiple files)",
#         type=["jpg", "jpeg", "png"],
#         accept_multiple_files=True,
#         key="multiple"
#     )

#     if uploaded_files:
#         results = []
#         for uploaded_file in uploaded_files:
#             # Display each uploaded image
#             image = Image.open(uploaded_file)
#             st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

#             # Simulate a progress bar for each image
#             with st.spinner(f"Analyzing {uploaded_file.name}..."):
#                 time.sleep(2)  # Simulate processing time

#             # Placeholder for model prediction (to be connected to the backend)
#             # Simulating a response for demonstration purposes
#             prediction = "Spirulina"  # Example: Replace with model's output
#             confidence = 0.95  # Example: Replace with model's confidence score

#             # Check against confidence threshold
#             if confidence * 100 >= confidence_threshold:
#                 result = f"**Classification:** {prediction} (Confidence: {confidence * 100:.2f}%)"
#             else:
#                 result = f"**Classification:** Not confident enough (Confidence: {confidence * 100:.2f}%)"

#             # Display the result for each image
#             st.write(f"### Result for {uploaded_file.name}:")
#             st.write(result)

#             # Add to history
#             st.session_state["history"].append({
#                 "image": uploaded_file.name,
#                 "prediction": prediction,
#                 "confidence": confidence * 100
#             })

#             # Append to results for report generation
#             results.append({
#                 "image": uploaded_file.name,
#                 "prediction": prediction,
#                 "confidence": confidence * 100
#             })

#         # Generate a downloadable report for all images
#         if results:
#             report = "Microalgae Morphology Detection Report\n"
#             report += "--------------------------------------\n"
#             for result in results:
#                 report += f"Image: {result['image']}\n"
#                 report += f"Classification: {result['prediction']}\n"
#                 report += f"Confidence: {result['confidence']:.2f}%\n"
#                 report += "--------------------------------------\n"
#             report_bytes = io.BytesIO(report.encode("utf-8"))
#             st.download_button(
#                 label="Download Report for All Images",
#                 data=report_bytes,
#                 file_name="microalgae_report_multiple.txt",
#                 mime="text/plain"
#             )

# # Display history
# if st.session_state["history"]:
#     st.write("### Prediction History:")
#     for i, entry in enumerate(st.session_state["history"]):
#         st.write(f"**{i+1}. Image:** {entry['image']} | **Prediction:** {entry['prediction']} | **Confidence:** {entry['confidence']:.2f}%")

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Load pre-trained model
model = tf.keras.models.load_model('spirulina_model.h5')

# Session state for prediction history
if 'history' not in st.session_state:
    st.session_state['history'] = []

def classify_image(image):
    # Preprocess image and predict
    img = image.resize((224, 224))  # Resize to model input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    confidence = prediction[0][0] * 100
    label = "Spirulina" if confidence > 50 else "Not Spirulina"
    return label, confidence

def main():
    st.title("Microalgae Morphology Detection")
    st.sidebar.title("About")
    st.sidebar.info(
        "This application uses a computer vision model to classify microscopic images of microalgae (Spirulina)."
    )

    # Confidence threshold slider
    confidence_threshold = st.slider("Set Confidence Threshold (%)", 50, 100, 90)

    # Upload options
    upload_option = st.radio("Upload Option", ["Single Image Upload", "Multiple Image Upload"])

    if upload_option == "Single Image Upload":
        uploaded_file = st.file_uploader("Upload a Microscopic Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            label, confidence = classify_image(image)
            st.write(f"**Prediction:** {label}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.session_state['history'].append({"Image": uploaded_file.name, "Prediction": label, "Confidence": confidence})

    elif upload_option == "Multiple Image Upload":
        uploaded_files = st.file_uploader("Upload Microscopic Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            results = []
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                label, confidence = classify_image(image)
                results.append({"Image": uploaded_file.name, "Prediction": label, "Confidence": confidence})
                st.image(image, caption=f"{uploaded_file.name} - {label} ({confidence:.2f}%)", use_column_width=True)
            st.session_state['history'].extend(results)

    # Downloadable report
    if st.session_state['history']:
        st.write("### Prediction History")
        history_df = pd.DataFrame(st.session_state['history'])
        st.dataframe(history_df)
        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Report", data=csv, file_name="classification_report.csv", mime="text/csv")

if __name__ == "__main__":
    main()
