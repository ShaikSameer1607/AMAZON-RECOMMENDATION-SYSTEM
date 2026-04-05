
import os
import requests
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

# Sample (you should replace with your pdf data later)
products = [
    {"product_id": 1, "product_name": "Sony Gaming Laptop Ultra (2025)"},
    {"product_id": 2, "product_name": "Waterproof TV by Acer"}
]

# Folder
os.makedirs("static/product_images", exist_ok=True)

def generate_product_image(product_name, product_id):
    prompt = f"Professional e-commerce product photo of {product_name}, white background, high quality"

    try:
        response = client.images.generate(
            model="gpt-image-1",   # ✅ UPDATED MODEL
            prompt=prompt,
            size="512x512"
        )

        image_base64 = response.data[0].b64_json

        filepath = f"static/product_images/{product_id}.png"

        with open(filepath, "wb") as f:
            f.write(base64.b64decode(image_base64))

        return filepath

    except Exception as e:
        print("Error:", e)
        return None
