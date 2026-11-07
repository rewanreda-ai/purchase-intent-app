import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Purchase Intent Predictor", page_icon="🛒")

THRESHOLD = 0.4


@st.cache_resource
def load_model():
    return joblib.load("purchase_intent_pipeline.joblib")


model = load_model()

st.title("🛒 Will This Visitor Buy?")
st.write("Enter a browsing session and get the purchase probability and a suggested action.")

with st.form("session_form"):
    st.subheader("Pages visited")
    c1, c2 = st.columns(2)
    administrative = c1.number_input("Account pages visited", 0, 30, 0, help="Login, account settings, etc.")
    administrative_dur = c2.number_input("Seconds on account pages", 0.0, 5000.0, 0.0)
    informational = c1.number_input("Info pages visited", 0, 30, 0, help="About, contact, terms, etc.")
    informational_dur = c2.number_input("Seconds on info pages", 0.0, 5000.0, 0.0)
    product = c1.number_input("Product pages visited", 0, 800, 10)
    product_dur = c2.number_input("Seconds on product pages", 0.0, 70000.0, 300.0)

    st.subheader("Behavior signals")
    bounce = st.slider("Bounce rate", 0.0, 0.2, 0.02, 0.005,
                       help="Share of visitors who leave after one page without doing anything.")
    exit_rate = st.slider("Exit rate", 0.0, 0.2, 0.04, 0.005,
                          help="How often the pages visited were the last page in a session.")
    page_values = st.number_input("Page value", 0.0, 400.0, 0.0,
                                  help="Average value of the pages visited before a purchase. Higher means closer to buying.")
    special_day = st.select_slider("Closeness to a special day", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], value=0.0)

    st.subheader("Visitor details")
    month = st.selectbox("Month", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], index=2)
    visitor = st.selectbox("Visitor type", ["Returning_Visitor", "New_Visitor", "Other"])
    weekend = st.checkbox("Weekend visit")
    os_ = st.selectbox("Operating system code", [str(i) for i in range(1, 9)], index=1,
                       help="Code used in the dataset.")
    browser = st.selectbox("Browser code", [str(i) for i in range(1, 14)], index=1,
                           help="Code used in the dataset.")
    region = st.selectbox("Region code", [str(i) for i in range(1, 10)],
                          help="Code used in the dataset.")
    traffic = st.selectbox("Traffic type code", [str(i) for i in range(1, 21)], index=1,
                           help="Code used in the dataset (source of the visit).")

    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([{
        "Administrative": administrative,
        "Administrative_Duration": administrative_dur,
        "Informational": informational,
        "Informational_Duration": informational_dur,
        "ProductRelated": product,
        "ProductRelated_Duration": product_dur,
        "BounceRates": bounce,
        "ExitRates": exit_rate,
        "PageValues": page_values,
        "SpecialDay": special_day,
        "Month": month,
        "OperatingSystems": os_,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic,
        "VisitorType": visitor,
        "Weekend": int(weekend),
    }])

    try:
        proba = model.predict_proba(row)[0, 1]
        st.metric("Purchase probability", f"{proba:.0%}")
        st.progress(float(proba))

        if proba >= 0.7:
            st.success("Very likely to buy. Suggested action: no discount needed, the visitor will probably buy anyway.")
        elif proba >= THRESHOLD:
            st.warning("Might buy. Suggested action: show a chat assistant or a small coupon to push the visit over the line.")
        else:
            st.info("Unlikely to buy. Suggested action: no action needed.")
    except Exception as e:
        st.error("Something went wrong with the input. Please check the values and try again.")
        st.caption(str(e))

st.caption("Decision-support tool built on the UCI Online Shoppers Purchasing Intention dataset. "
           "Probabilities are model estimates, not guarantees. "
           "Category codes (browser, region, traffic type) are the dataset's own codes.")