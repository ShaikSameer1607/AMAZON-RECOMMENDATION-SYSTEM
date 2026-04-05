import os
from flask import Flask, jsonify, render_template, request
import pandas as pd
from product_generator import MassiveProductNameGenerator

app = Flask(__name__)
name_gen = MassiveProductNameGenerator()

# Load Data
try:
    df = pd.read_json("final_recommendations.json")
    df['user_id'] = df['user_id'].astype(str)
except Exception as e:
    print(f"Error: {e}")
    df = pd.DataFrame()

def get_realistic_price(category, product_id):
    # Deterministic seed based on product_id so price stays same on refresh
    seed = abs(hash(str(product_id)))
    
    # Realistic Price Brackets (in INR)
    price_map = {
        "Electronics": (25000, 150000), # Laptops/Phones
        "Books": (299, 1499),           # Standard Books
        "Home": (1500, 25000),          # Furniture/Appliances
        "Sports": (499, 12000)          # Gear/Apparel
    }
    
    low, high = price_map.get(category, (1000, 5000))
    return (seed % (high - low)) + low

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/users")
def get_users():
    if not df.empty:
        # Pulling ALL unique users from the 2.3M records
        return jsonify(sorted(df['user_id'].unique().tolist()))
    return jsonify([])

@app.route("/api/recommendations")
def get_recommendations():
    user_id = request.args.get('user_id')
    user_data = df[df['user_id'] == str(user_id)] if user_id else pd.DataFrame()

    if user_data.empty:
        display_data = df.sort_values(by="rating", ascending=False).head(50).to_dict(orient="records")
    else:
        # Show all recommendations for this user (usually 10-50 items from Spark)
        display_data = user_data.to_dict(orient="records")

    placeholders = {
        "Electronics": "https://images.unsplash.com/photo-1498049794561-7780e7231661?w=500",
        "Books": "https://images.unsplash.com/photo-1495446815901-a7297e633e8d?w=500",
        "Sports": "https://images.unsplash.com/photo-1461896836934-ffe607ba8211?w=500",
        "Home": "https://images.unsplash.com/photo-1556228578-0d85b1a4d571?w=500"
    }

    for item in display_data:
        # 1. Generate Realistic Name
        item['product_name'] = name_gen.generate_name(item['product_id'], user_id)
        
        # 2. Categorize
        n = item['product_name'].lower()
        if any(x in n for x in ["laptop", "phone", "tv", "camera"]): cat = "Electronics"
        elif "book" in n: cat = "Books"
        elif any(x in n for x in ["gym", "shoe", "sport"]): cat = "Sports"
        else: cat = "Home"
        
        item['category'] = cat
        
        # 3. Apply Realistic Price
        item['price'] = get_realistic_price(cat, item['product_id'])
        
        # 4. Image Logic
        img_path = f"static/product_images/{item['product_id']}.png"
        item['image_url'] = f"/{img_path}" if os.path.exists(img_path) else placeholders.get(cat)

    return jsonify(display_data)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
