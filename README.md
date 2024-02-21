Media Design School
barry@xander.co.nz

Working Demo: https://fruitclassifier-barrymoana.streamlit.app/

### Importing Required Libraries
The script begins by importing necessary Python libraries:

`streamlit` for building the web app interface.
`PIL` (Python Imaging Library) for image processing tasks.
`numpy` for numerical operations, particularly for handling image data.
`matplotlib.pyplot` for plotting images and annotations.
`roboflow` for accessing and utilizing the Roboflow AI platform.
`tempfile` for creating temporary files, useful for handling uploaded images.


### Initializing the Roboflow Model
Initializes the Roboflow model with an API key, specifies the project and version, and sets the confidence and overlap thresholds for the model predictions.

```
# Initialize Roboflow model
rf = roboflow.Roboflow(api_key="xEWche4yxaVUUD2EqXKo")
project = rf.workspace().project("fruit-detection-v3")
model = project.version("4").model

# Set model confidence and overlap threshold
model.confidence = 75
model.overlap = 25
```

### Defining Color and Text Mappings for Annotations
Sets up dictionaries for mapping fruit classes to specific colors for drawing bounding boxes and for defining text colors for labels. Additionally, it specifies a default text color and font size for annotations.

```
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
```

### Function to Annotate Images
Defines a function that takes an image and the prediction results to annotate the image with bounding boxes and labels. It uses matplotlib for drawing.

```
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

        plt.text(label_pos_x, label_pos_y, f"{class_label} ({confidence}%)", color=label_text_color, fontsize=text_font_size,
                 bbox=dict(facecolor=poly_color, alpha=0.5, edgecolor='none'))

    plt.axis('off')
    return fig
```

### Streamlit Web Interface Setup
Sets up the Streamlit web interface, including a title and a file uploader for users to upload images. It then processes the uploaded images, displays them, and uses the Roboflow model to make predictions and annotate the images based on those predictions.

```
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
```


















