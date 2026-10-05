import torch

from PIL import Image

from model import create_model

from transforms import val_transform

model = create_model()
model.load_state_dict(
    torch.load(
        "C:\\Projects\\plant-disease-capstone\\models\\best_model.pth",
        map_location=torch.device("cpu")
    )
)
#model.to("cuda" if torch.cuda.is_available() else "cpu")
map_location = torch.device("cpu")
model.eval()



def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = val_transform(image)

    image = image.unsqueeze(0)

    image = image.to(torch.device("cpu")) 
    with torch.no_grad():

        outputs = model(image)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        confidence, pred = torch.max(
            probs,
            1
        )

    class_names = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
    ]

    return (
        f"disease: {class_names[pred.item()]}",
        f"confidence: {confidence.item()}"
    )

recommendations = {

    "Tomato___Bacterial_spot":
    """
    Remove heavily infected leaves.
    Avoid overhead irrigation and improve air circulation.
    Follow local agricultural guidance for bactericide use.
    """,

    "Tomato___Early_blight":
    """
    Remove infected leaves and plant debris.
    Apply an appropriate fungicide if recommended locally.
    Practice crop rotation and avoid excessive leaf wetness.
    """,

    "Tomato___Late_blight":
    """
    Remove infected plant material immediately.
    Improve drainage and airflow around plants.
    Follow local recommendations for protective fungicide applications.
    """,

    "Tomato___Leaf_Mold":
    """
    Reduce humidity around plants.
    Increase ventilation in greenhouse or covered environments.
    Remove infected leaves and monitor disease spread.
    """,

    "Tomato___Septoria_leaf_spot":
    """
    Remove infected foliage.
    Avoid watering leaves directly.
    Maintain proper plant spacing and sanitation practices.
    """,

    "Tomato___Spider_mites Two-spotted_spider_mite":
    """
    Inspect the undersides of leaves regularly.
    Increase humidity when appropriate and remove heavily affected leaves.
    Consider integrated pest management practices.
    """,

    "Tomato___Target_Spot":
    """
    Remove infected leaves and fallen debris.
    Improve airflow around plants.
    Monitor disease progression and follow local management guidelines.
    """,

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
    """
    Remove severely affected plants.
    Control whitefly populations, which commonly spread this virus.
    Use resistant varieties when available.
    """,

    "Tomato___Tomato_mosaic_virus":
    """
    Remove infected plants to reduce spread.
    Sanitize tools and avoid handling healthy plants after infected ones.
    Use certified disease-free planting material.
    """,

    "Tomato___healthy":
    """
    Plant appears healthy.
    Continue regular monitoring, proper irrigation, balanced fertilization,
    and preventive disease management practices.
    """
}

disease, confidence = predict_image("C:\\Projects\\plant-disease-capstone\\src\\21fe0741-4ff6-4f8f-8969-6b8bdeea58fa___PSU_CG 2278.JPG")

# Extract the actual class name by removing the "disease: " prefix
clean_disease = disease.replace("disease: ", "").strip()

print(disease,confidence,"recommendation:",recommendations[clean_disease])

