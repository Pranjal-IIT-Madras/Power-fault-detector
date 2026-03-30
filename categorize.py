"""
Smart Expense Categorizer
Classifies bank statement transactions using NLP (TF-IDF + Logistic Regression)
Supports CSV, Excel, and image inputs (via OCR)
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import re
import joblib
import pytesseract
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Category keywords for rule-augmented training data ──────────────────────
SEED_DATA = {
    "Food & Dining": [
        "zomato", "swiggy", "dominos", "pizza", "burger", "restaurant", "cafe",
        "food", "dining", "mcdonalds", "kfc", "subway", "starbucks", "barbeque",
        "hotel food", "biryani", "mess", "canteen", "snacks", "beverage"
    ],
    "Transport": [
        "uber", "ola", "rapido", "metro", "bus ticket", "railway", "irctc",
        "petrol", "fuel", "diesel", "parking", "toll", "cab", "auto", "rickshaw",
        "bike rental", "flight", "indigo", "air india", "spicejet"
    ],
    "Shopping": [
        "amazon", "flipkart", "myntra", "ajio", "meesho", "nykaa", "shoppers stop",
        "clothing", "shoes", "electronics", "mobile", "laptop", "gadgets",
        "apparel", "fashion", "retail", "mall", "store", "purchase"
    ],
    "Bills & Utilities": [
        "electricity", "water bill", "gas", "broadband", "airtel", "jio", "bsnl",
        "vi vodafone", "recharge", "dth", "tata sky", "netflix", "hotstar",
        "spotify", "youtube premium", "rent", "maintenance", "society"
    ],
    "Health": [
        "pharmacy", "medical", "hospital", "clinic", "doctor", "medicine",
        "apollo", "1mg", "netmeds", "health", "lab test", "diagnostic",
        "gym", "fitness", "cult fit", "yoga", "wellness"
    ],
    "Education": [
        "udemy", "coursera", "books", "stationery", "tuition", "fees",
        "college", "university", "exam", "course", "study", "library",
        "byju", "unacademy", "skill", "certification"
    ],
    "Entertainment": [
        "movie", "pvr", "inox", "cinema", "concert", "event", "bookmyshow",
        "gaming", "steam", "playstation", "xbox", "game", "pub", "bar",
        "club", "party", "outing", "trip", "picnic"
    ],
    "Finance & Banking": [
        "emi", "loan", "insurance", "lic", "mutual fund", "sip", "investment",
        "fd", "fixed deposit", "credit card", "payment", "transfer", "neft",
        "imps", "upi", "bank charge", "interest", "dividend"
    ],
    "Groceries": [
        "bigbasket", "blinkit", "grofers", "zepto", "dmart", "reliance fresh",
        "supermarket", "kirana", "vegetables", "fruits", "milk", "dairy",
        "grocery", "mart", "provisions", "household"
    ],
    "Others": [
        "cash withdrawal", "atm", "miscellaneous", "other", "unknown", "general"
    ]
}


def build_training_data():
    """Generate labeled training data from seed keywords."""
    texts, labels = [], []
    for category, keywords in SEED_DATA.items():
        for kw in keywords:
            texts.append(kw)
            labels.append(category)
            # Add slight variations
            texts.append(f"payment to {kw}")
            labels.append(category)
            texts.append(f"{kw} purchase")
            labels.append(category)
    return texts, labels


def train_model():
    """Train and return the NLP pipeline."""
    print("🔧 Training NLP model...")
    texts, labels = build_training_data()
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            analyzer="char_wb",
            min_df=1
        )),
        ("clf", LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs",
                                   multi_class="multinomial"))
    ])
    pipeline.fit(texts, labels)
    print("✅ Model trained successfully.")
    return pipeline


def preprocess(text: str) -> str:
    """Lowercase and clean transaction description."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_csv_excel(filepath: str) -> pd.DataFrame:
    """Load CSV or Excel file and extract transactions."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    # Try to auto-detect description column
    desc_col = None
    for col in df.columns:
        if any(k in col.lower() for k in ["desc", "narr", "detail", "particular",
                                            "remark", "transaction", "info"]):
            desc_col = col
            break
    if desc_col is None:
        # Fall back to first string column
        for col in df.columns:
            if df[col].dtype == object:
                desc_col = col
                break

    if desc_col is None:
        raise ValueError("Could not detect a description column. "
                         "Please ensure your file has a column named "
                         "'Description', 'Narration', or similar.")

    print(f"📋 Using column '{desc_col}' as transaction description.")
    df["_description"] = df[desc_col].fillna("").astype(str)
    return df, desc_col


def read_image(filepath: str) -> pd.DataFrame:
    """OCR an image of a bank statement and extract transaction lines."""
    print(f"🖼️  Running OCR on image: {filepath}")
    img = Image.open(filepath)
    raw_text = pytesseract.image_to_string(img)

    lines = [l.strip() for l in raw_text.split("\n") if len(l.strip()) > 5]
    # Filter lines that look like transactions (contain letters + amount pattern)
    txn_lines = []
    for line in lines:
        if re.search(r"[a-zA-Z]{3,}", line) and re.search(r"\d", line):
            txn_lines.append(line)

    if not txn_lines:
        txn_lines = lines  # fallback: use all non-empty lines

    df = pd.DataFrame({"Description": txn_lines})
    df["_description"] = df["Description"]
    return df, "Description"


def categorize(df: pd.DataFrame, model, desc_col: str) -> pd.DataFrame:
    """Apply model to categorize each transaction."""
    cleaned = df["_description"].apply(preprocess)
    df["Category"] = model.predict(cleaned)
    proba = model.predict_proba(cleaned)
    df["Confidence"] = np.max(proba, axis=1).round(2)
    return df


def summarize(df: pd.DataFrame):
    """Print a category-wise expense summary."""
    print("\n" + "═" * 50)
    print("  📊 EXPENSE CATEGORY SUMMARY")
    print("═" * 50)

    # Try to detect amount column
    amt_col = None
    for col in df.columns:
        if any(k in col.lower() for k in ["amount", "debit", "credit", "amt", "sum"]):
            amt_col = col
            break

    if amt_col:
        try:
            df[amt_col] = pd.to_numeric(
                df[amt_col].astype(str).str.replace(",", "").str.extract(r"([\d.]+)")[0],
                errors="coerce"
            )
            summary = df.groupby("Category")[amt_col].sum().sort_values(ascending=False)
            for cat, total in summary.items():
                if not np.isnan(total):
                    print(f"  {cat:<25} ₹{total:>10,.2f}")
        except Exception:
            pass
    else:
        summary = df["Category"].value_counts()
        for cat, count in summary.items():
            print(f"  {cat:<25} {count:>5} transactions")

    print("═" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Smart Expense Categorizer — classify bank statement transactions"
    )
    parser.add_argument("input", help="Path to CSV, Excel (.xlsx), or image file")
    parser.add_argument("-o", "--output", default="categorized_expenses.csv",
                        help="Output CSV file path (default: categorized_expenses.csv)")
    parser.add_argument("--model", default=None,
                        help="Path to saved model (.pkl). Trains fresh if not provided.")
    parser.add_argument("--save-model", default="expense_model.pkl",
                        help="Save trained model to this path")
    args = parser.parse_args()

    # Load or train model
    if args.model and os.path.exists(args.model):
        print(f"📦 Loading model from {args.model}")
        model = joblib.load(args.model)
    else:
        model = train_model()
        joblib.dump(model, args.save_model)
        print(f"💾 Model saved to {args.save_model}")

    # Read input
    ext = os.path.splitext(args.input)[1].lower()
    if ext in [".csv", ".xlsx", ".xls"]:
        df, desc_col = read_csv_excel(args.input)
    elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        df, desc_col = read_image(args.input)
    else:
        print(f"❌ Unsupported file type: {ext}")
        sys.exit(1)

    print(f"📂 Loaded {len(df)} transactions from '{args.input}'")

    # Categorize
    df = categorize(df, model, desc_col)

    # Drop internal column
    df.drop(columns=["_description"], inplace=True, errors="ignore")

    # Save output
    df.to_csv(args.output, index=False)
    print(f"\n✅ Results saved to: {args.output}")

    # Summary
    summarize(df)

    print(f"\n📄 Preview (first 10 rows):")
    print(df[[desc_col, "Category", "Confidence"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
