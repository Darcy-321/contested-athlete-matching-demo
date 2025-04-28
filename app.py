import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# Load your mock athlete data
@st.cache_data
def load_data():
    return pd.read_csv("mock_athlete_dataset_full_v2.csv")

# Load MiniLM model and tokenizer
@st.cache_resource
def load_minilm_model():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return tokenizer, model

tokenizer, model = load_minilm_model()
df_athletes = load_data()

st.title("Contested: Athlete–Brand Matching Engine Demo")

# Helpers
def get_embeddings(text_list):
    tokens = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

def auto_generate_creative_direction(product_service, brand_personality, causes_cared_about, campaign_goals):
    personality_text = ", ".join(brand_personality) if brand_personality else "dynamic"
    cause_text = ", ".join(causes_cared_about) if causes_cared_about else "key social causes"
    goals_text = ", ".join(campaign_goals) if campaign_goals else "brand building"
    creative_direction = (
        f"We're seeking athletes who can bring the {personality_text.lower()} spirit of our {product_service.lower()} to life. "
        f"Ideally, they care about causes like {cause_text.lower()} and can help us achieve goals like {goals_text.lower()}. "
        "Authentic storytelling and relatable content are important to us."
    )
    return creative_direction.strip()

def auto_suggest_athlete_narratives(brand_personality, campaign_goals):
    suggestions = set()
    if "Empowering" in brand_personality:
        suggestions.update(["Mentor", "Trailblazer"])
    if "Funny" in brand_personality:
        suggestions.update(["Goofy", "Relatable"])
    if "High-Performance" in brand_personality:
        suggestions.update(["Trailblazer", "Captain / Team Leader"])
    if "Ethical" in brand_personality:
        suggestions.update(["Community Leader", "Mentor"])
    if "Brand Awareness" in campaign_goals:
        suggestions.update(["Rising Star", "Trailblazer"])
    if "Trust & Relatability" in campaign_goals:
        suggestions.update(["Mentor", "Big Sister/Brother"])
    return list(suggestions)

def generate_fit_explanation(row):
    reasons = []
    if row.get("Region Match"):
        reasons.append("matches your target region")
    if row.get("Archetype Match"):
        reasons.append("fits your ideal athlete narrative")
    if row.get("Cause Match"):
        reasons.append("aligns with causes you care about")
    if row.get("Gender Match"):
        reasons.append("matches your preferred gender")
    if row.get("Sport Match"):
        reasons.append("plays your preferred sport")
    if row.get("Audience Match"):
        reasons.append("has a matching audience demographic")
    if row.get("High Engagement"):
        reasons.append("has strong social engagement")
    if row.get("Recently Active"):
        reasons.append("is recently active on social media")
    return "; ".join(reasons)

def checkbox_grid(options, cols=3):
    selected = []
    columns = st.columns(cols)
    for idx, option in enumerate(options):
        if columns[idx % cols].checkbox(option):
            selected.append(option)
    return selected

# 1. Athlete Profiles
st.subheader("1. Athlete Profiles")
st.dataframe(df_athletes)

# 2. Brand Inputs
st.subheader("2. Enter Brand Campaign Info")

brand_name = st.text_input("Brand Name")
product_service = st.text_input("Product / Service Type")

industry_sector = st.selectbox("Industry/Sector", [
    "Select an option...",
    "Apparel", "Beauty & Personal Care", "Education/EdTech", "Energy/Sustainability",
    "Finance & Insurance", "Food & Beverage", "Health & Wellness", "Local Retail / Small Business",
    "Nonprofit / Cause-based", "Tech / Consumer Electronics", "Travel & Hospitality", "Other"
])

product_type = st.selectbox("Type (Product or Service)", [
    "Select an option...", "Product", "Service"
])

target_region = st.selectbox("Target Region", [
    "Select an option...", "Southeast US", "Midwest", "West Coast"
])

st.markdown("### Brand Personality")
brand_personality = checkbox_grid([
    "Aspirational", "Bold", "Educational", "Empowering", "Ethical", "Friendly",
    "Funny", "High-Performance", "Reliable", "Warm", "Youthful"
])

st.markdown("### Causes You Care About")
causes_cared_about = checkbox_grid([
    "Anti-Bullying", "Disability Inclusion", "Education", "Faith / Spirituality", "Family",
    "Female Empowerment", "Financial Literacy", "Gender Equality", "Immigrant Rights",
    "LGBTQ+ Inclusion", "Local Community Empowerment", "Mental Health",
    "NIL Rights / Athlete Advocacy", "Nutrition / Wellness", "Racial Justice",
    "Sustainability", "Veteran Support", "Youth Mentorship"
])

st.markdown("### Target Customers")
target_customers = checkbox_grid([
    "Gen Z", "Millennials", "Parents", "Athletes", "Fitness Enthusiasts"
])

st.markdown("### Campaign Goals")
campaign_goals = checkbox_grid([
    "Brand Awareness", "Brand Lift", "Cause Alignment", "Community Engagement",
    "Lead Generation / Conversion", "Trust & Relatability", "User-Generated Content"
])

budget_range = st.text_input("Budget Range (e.g., $5,000-$10,000)")

# Auto-generate Creative Direction
default_creative_direction = auto_generate_creative_direction(
    product_service, brand_personality, causes_cared_about, campaign_goals
)
creative_notes = st.text_area("Creative Direction", value=default_creative_direction)

# Auto-suggest Ideal Athlete Narrative
suggested_archetypes = auto_suggest_athlete_narratives(brand_personality, campaign_goals)

st.markdown("### Ideal Athlete Narrative")
all_archetypes = [
    "Big Sister/Brother", "Captain / Team Leader", "Comeback Kid", "Community Leader",
    "Faith-based", "First-Gen Success", "Late Bloomer", "Mentor", "Rising Star",
    "Trailblazer", "Underdog"
]
columns = st.columns(3)
ideal_archetypes = []
for idx, archetype in enumerate(all_archetypes):
    default = archetype in suggested_archetypes
    if columns[idx % 3].checkbox(archetype, value=default):
        ideal_archetypes.append(archetype)

target_gender = st.selectbox("Preferred Athlete Gender", [
    "Select an option...", "Female", "Male", "Any"
])

target_sport = st.selectbox("Preferred Sport", [
    "Select an option...", "Baseball", "Basketball", "Football", "Golf", "Lacrosse", "Rugby",
    "Soccer", "Softball", "Swimming", "Tennis", "Track & Field", "Volleyball", "Wrestling", "Other"
])

# 3. Run Matching
if st.button("Run Matching Engine"):
    st.subheader("3. Data Preprocessing")

    structured_features = {
        "target_region": target_region,
        "ideal_archetypes": ideal_archetypes,
        "causes_cared_about": causes_cared_about,
        "brand_personality": brand_personality,
        "campaign_goals": campaign_goals,
        "budget_range": budget_range,
        "target_gender": target_gender,
        "target_sport": target_sport,
    }
    st.json(structured_features)

    st.markdown("**Narrative Embedding Text (for BERT/BiLSTM):**")
    brand_narrative = f"{brand_name} offers {product_service} targeting {', '.join(target_customers)}. {creative_notes}"
    st.write(brand_narrative)

    st.subheader("4. Matching Scores")

    today = datetime.today()

    weight_engagement = 1
    weight_audience = 1
    weight_last_active = 1

    if "Trust & Relatability" in campaign_goals:
        weight_engagement += 1
    if "Brand Awareness" in campaign_goals:
        weight_audience += 1
    if "Community Engagement" in campaign_goals:
        weight_last_active += 1

    def structured_match(row):
        score = 0
        row["Region Match"] = target_region in str(row.get("Playing Location", ""))
        row["Archetype Match"] = any(archetype in str(row.get("Narrative Archetypes", "")) for archetype in ideal_archetypes)
        row["Cause Match"] = any(cause in str(row.get("Causes / Values", "")) for cause in causes_cared_about)
        row["Gender Match"] = target_gender != "Any" and target_gender == str(row.get("Gender", ""))
        row["Sport Match"] = target_sport and target_sport.lower() in str(row.get("Sport", "")).lower()
        row["High Engagement"] = float(row.get("Engagement Rate (%)", 0)) >= 5.0
        last_active_days = (today - pd.to_datetime(row.get("Last Active Date"))).days
        row["Recently Active"] = last_active_days <= 14
        row["Audience Match"] = any(demo in str(row.get("Audience Demographics", "")) for demo in target_customers)

        score += int(row["Region Match"])
        score += int(row["Archetype Match"])
        score += int(row["Cause Match"])
        score += int(row["Gender Match"])
        score += int(bool(row["Sport Match"]))
        score += int(row["High Engagement"]) * weight_engagement
        score += int(row["Recently Active"]) * weight_last_active
        score += int(row["Audience Match"]) * weight_audience

        row["Structured Match Score"] = score
        return row

    df_athletes = df_athletes.apply(structured_match, axis=1)

    # Narrative similarity
    athlete_narratives = (
        df_athletes["Narrative Archetypes"].fillna("") + " " +
        df_athletes["Content Strengths"].fillna("") + " " +
        df_athletes["Core Identity Values"].fillna("") + " " +
        df_athletes["Defining Moments"].fillna("") + " " +
        df_athletes["Style Notes"].fillna("")
    )

    brand_embedding = get_embeddings([brand_narrative])
    athlete_embeddings = get_embeddings(athlete_narratives.tolist())

    similarities = cosine_similarity(brand_embedding, athlete_embeddings)[0]
    df_athletes["Narrative Compatibility Score"] = np.round(similarities, 2)

    df_athletes["Combined Score"] = df_athletes["Structured Match Score"] * 0.6 + df_athletes["Narrative Compatibility Score"] * 5 * 0.4
    df_athletes["Fit Explanation"] = df_athletes.apply(generate_fit_explanation, axis=1)

    st.subheader("5. Ranked Athlete Matches")
    ranked_athletes = df_athletes.sort_values("Combined Score", ascending=False).reset_index(drop=True)

    st.dataframe(ranked_athletes[[
        "Full Name", "Structured Match Score", "Narrative Compatibility Score",
        "Combined Score", "Fit Explanation"
    ]])
