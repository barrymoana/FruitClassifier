import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import roboflow
import tempfile

# Initialize Roboflow model
rf = roboflow.Roboflow(api_key="xEWche4yxaVUUD2EqXKo")
project = rf.workspace().project("fruit-detection-v3")
model = project.version("4").model

# Set model confidence and overlap threshold
model.confidence = 75
model.overlap = 25

# Color and text maps
color_map = {
    'fresh_apple': 'lime',
    'fresh_orange': 'lime',
    'rotten_banana': 'red',
}
text_color_map = {
    'fresh_apple': 'black',
    'fresh_orange': 'black',
    'rotten_banana': 'white',
}
default_text_color = 'white'
text_font_size = 7

# Function to plot and annotate image
def annotate_image(image, predictions):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)

    for det in predictions['predictions']:
        class_label = det['class']
        confidence = round(det['confidence'] * 100, 2)
        points = det['points']
        poly = np.array([[point['x'], point['y']] for point in points])
        poly_color = color_map.get(class_label, 'white')
        poly_path = plt.Polygon(poly, edgecolor=poly_color, facecolor='none', linewidth=1)
        ax.add_patch(poly_path)
        label_pos_x = poly[0, 0]
        label_pos_y = poly[0, 1]
        label_text_color = text_color_map.get(class_label, default_text_color)

        # Use the smaller font size for text labels
        plt.text(label_pos_x, label_pos_y, f"{class_label} ({confidence}%)", color=label_text_color, fontsize=text_font_size,
                 bbox=dict(facecolor=poly_color, alpha=0.5, edgecolor='none'))

    plt.axis('off')  # Moved outside of the loop
    return fig

# Streamlit interface
st.title('Fruit Detection App')

# File uploader allows user to add their own image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())

    image_path = temp_file.name

    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Perform prediction
    prediction = model.predict(image_path)

    # Annotate image
    fig = annotate_image(image, prediction.json())
    st.pyplot(fig)
