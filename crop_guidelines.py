def get_guidelines(crop, soil_type=None, ph=None):

    tips = {
        "rice": "💧 Needs high water and warm climate. Best for clayey soil.",
        "wheat": "🌾 Requires cool climate and moderate water. Suitable for loamy soil.",
        "maize": "🌽 Needs well-drained soil and good sunlight.",
        "cotton": "🌿 Requires warm climate and low rainfall.",
        "sugarcane": "🍬 Needs high water and fertile soil.",
        "millet": "🌾 Grows well in dry and sandy soil.",
        "groundnut": "🥜 Requires sandy soil and low rainfall.",
        "barley": "🌾 Grows well in alkaline soil and low water.",
        "potato": "🥔 Prefers cool climate and slightly acidic soil.",
        "tea": "🍃 Requires acidic soil and high rainfall.",
        "vegetables": "🥕 Require fertile loamy soil and regular watering.",
        "pulses": "🌱 Grow well in well-drained soil with low nitrogen."
    }

    # ----------------------------
    # CASE 1: FERTILIZER OUTPUT
    # ----------------------------
    if isinstance(crop, str) and "-" in crop:
        try:
            n, p, k = crop.split("-")
        except:
            return "❌ Invalid fertilizer format."

        return f"""
🧪 **Fertilizer Recommendation: {crop}**

• Nitrogen (N): {n}% → 🌿 Leaf growth  
• Phosphorus (P): {p}% → 🌱 Root development  
• Potassium (K): {k}% → 🌾 Overall plant health  

👉 Apply this fertilizer to balance soil nutrients.

📌 **Best Practices:**
• Apply in split doses  
• Mix with organic compost  
• Avoid overuse to protect soil health  
"""

    # ----------------------------
    # CASE 2: CROP GUIDELINES
    # ----------------------------
    crop_lower = str(crop).strip().lower()

    base_tip = tips.get(crop_lower, "🌱 Follow general farming practices.")

    extra = ""

    # ----------------------------
    # SOIL TYPE BASED SUGGESTION
    # ----------------------------
    if soil_type:
        soil_type = soil_type.strip()

        if soil_type == "Clayey":
            extra += "\n• 🌍 Clayey soil retains water → suitable for water-intensive crops."
        elif soil_type == "Sandy":
            extra += "\n• 🌍 Sandy soil drains quickly → add compost and irrigate frequently."
        elif soil_type == "Loamy":
            extra += "\n• 🌍 Loamy soil is ideal → supports most crops."

    # ----------------------------
    # pH BASED SUGGESTION
    # ----------------------------
    if ph is not None:
        try:
            ph = float(ph)

            if ph < 6:
                extra += "\n• ⚠ Soil is acidic → add lime to balance pH."
            elif ph > 7.5:
                extra += "\n• ⚠ Soil is alkaline → add gypsum."
            else:
                extra += "\n• ✅ Soil pH is optimal for crop growth."
        except:
            pass

    # ----------------------------
    # DEFAULT EXTRA (FIX)
    # ----------------------------
    if extra.strip() == "":
        extra = "\n• Maintain proper irrigation and nutrient balance."

    # ----------------------------
    # FINAL OUTPUT
    # ----------------------------
    return f"""
🌾 **Crop: {crop.capitalize()}**

{base_tip}

📊 **Soil & Condition Tips:**
{extra}
"""