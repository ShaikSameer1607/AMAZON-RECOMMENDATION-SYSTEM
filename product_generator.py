
import random

class MassiveProductNameGenerator:
    def __init__(self):
        self.brands = [ "Apple","Samsung","Sony","Dell","HP","Lenovo","Asus","Acer","MSI","Razer","LG","Panasonic","Bose","JBL","Nike","Adidas" ]

        self.categories = [
            "Laptop","Smartphone","Headphones","Monitor","Tablet","Camera",
            "Smart Watch","Printer","TV","Speaker","Keyboard","Mouse"
        ]

        self.modifiers = [
            "Pro","Max","Ultra","Lite","Plus","Elite","Premium","Advanced",
            "Gaming","Business","Portable","Wireless","Smart","Compact"
        ]

        self.colors = [
            "Black","Silver","Blue","Red","White","Gold","Space Gray"
        ]

        self.features = [
            "Touch Screen","Noise Cancelling","Fast Charging","Bluetooth",
            "Waterproof","4K","HDR","WiFi 6","RGB Lighting"
        ]

        self.sizes = [
            "Mini","Compact","Standard","Large","15 Inch","27 Inch"
        ]

        self.materials = [
            "Aluminum","Plastic","Carbon Fiber","Steel","Glass"
        ]

    def generate_name(self, product_id, user_id=None):
        # 🔥 Better randomness
        seed_input = f"{product_id}" if user_id is None else f"{user_id}_{product_id}"
        random.seed(abs(hash(seed_input)) % (10**8))

        # 🔥 MANY templates (this is key)
        templates = [

            lambda: f"{random.choice(self.brands)} {random.choice(self.categories)} {random.choice(self.modifiers)}",

            lambda: f"{random.choice(self.brands)} {random.choice(self.categories)} - {random.choice(self.colors)}",

            lambda: f"{random.choice(self.categories)} {random.choice(self.modifiers)} {random.choice(self.colors)}",

            lambda: f"{random.choice(self.features)} {random.choice(self.categories)} by {random.choice(self.brands)}",

            lambda: f"{random.choice(self.brands)} {random.choice(self.modifiers)} {random.choice(self.categories)} ({random.choice(self.sizes)})",

            lambda: f"{random.choice(self.colors)} {random.choice(self.materials)} {random.choice(self.categories)}",

            lambda: f"{random.choice(self.categories)} for {random.choice(['Gaming','Office','Travel','Home'])}",

            lambda: f"{random.choice(['Premium','Ultimate','Deluxe'])} {random.choice(self.categories)} by {random.choice(self.brands)}",

            lambda: f"{random.choice(self.brands)} {random.choice(self.categories)} with {random.choice(self.features)}",

            lambda: f"{random.choice(self.features)} {random.choice(self.colors)} {random.choice(self.categories)}"
        ]

        name = random.choice(templates)()

        # 🔥 EXTRA VARIATION
        if random.random() > 0.6:
            name += f" - {random.choice(self.colors)}"

        if random.random() > 0.7:
            name += f" ({random.choice(['2024','2025','Latest'])})"

        if random.random() > 0.9:
            name += f" - {random.choice(['Best Seller','Amazon Choice'])}"

        return name
