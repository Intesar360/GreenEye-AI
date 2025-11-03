from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os
import uuid

# ===============================
# CONFIG
# ===============================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # base directory for relative paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")

LEAF_MODEL_PATH = os.path.join(MODEL_DIR, "leaf_detector.pth")
DISEASE_MODEL_PATH = os.path.join(MODEL_DIR, "best_cpu_model.pth")
SPECIFIC_DISEASE_MODEL_PATH = os.path.join(MODEL_DIR, "disease_stage2_best_model.pth")

DEVICE = torch.device("cpu")
LEAF_CLASS_NAMES = ["Leaf", "Not Leaf"]
DISEASE_CLASS_NAMES = ["Dry", "Healthy", "Unhealthy"]

SPECIFIC_DISEASE_CLASSES = [
    "Anthracnose","Anthrax_Leaf","Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust",
    "Bacterial_Blight","Bacterial_Canker","Bituminous_Leaf","Black_Spot","Cherry_including_sour___Powdery_mildew",
    "Curl_Leaf","Curl_Virus","Cutting_Weevil","Deficiency_Leaf","Die_Back",
    "Entomosporium_Leaf_Spot_on_woody_ornamentals","Felt_Leaf","Fungal_Leaf_Spot","Gall_Midge","Leaf_Blight",
    "Leaf_Gall","Leaf_Holes","Leaf_blight_Litchi_leaf_diseases","Litchi_algal_spot_in_non-direct_sunlight",
    "Litchi_anthracnose_on_cloudy_day","Litchi_leaf_mites_in_direct_sunlight","Litchi_mayetiola_after_raining",
    "Pepper__bell___Bacterial_spot","Potato___Early_blight","Potato___Late_blight","Powdery_Mildew",
    "Sooty_Mould","Spider_Mites","Tomato__Tomato_YellowLeaf__Curl_Virus","Tomato___Bacterial_spot",
    "Tomato___Early_blight","Tomato___Late_blight","Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot",
    "Tomato___Target_Spot","Tomato___Tomato_mosaic_virus"
]

THRESHOLD_LEAF = 0.8
THRESHOLD_UNHEALTHY = 0.6

# ===============================
# APP SETUP
# ===============================
app = Flask(__name__)
app.secret_key = "super_secret_123"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

db = SQLAlchemy(app)

# ===============================
# DATABASE MODELS
# ===============================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    predictions = db.relationship("Prediction", backref="user", lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(200))
    leaf = db.Column(db.String(50))
    health = db.Column(db.String(50))
    disease = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

# ===============================
# MODEL LOADING
# ===============================
def load_leaf_model():
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(LEAF_CLASS_NAMES))
    model.load_state_dict(torch.load(LEAF_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def load_disease_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, len(DISEASE_CLASS_NAMES)))
    model.load_state_dict(torch.load(DISEASE_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def load_specific_disease_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, len(SPECIFIC_DISEASE_CLASSES)))
    model.load_state_dict(torch.load(SPECIFIC_DISEASE_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

leaf_model = load_leaf_model()
disease_model = load_disease_model()
specific_disease_model = load_specific_disease_model()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ===============================
# ROUTES
# ===============================
@app.route("/")
def home():
    if "user_id" in session:
        if session.get("is_admin"):
            return redirect(url_for("admin_dashboard"))
        else:
            return redirect(url_for("user_dashboard"))
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])
        db.session.add(User(email=email, password=password))
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["is_admin"] = user.is_admin
            return redirect(url_for("admin_dashboard" if user.is_admin else "user_dashboard"))
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/user_dashboard")
def user_dashboard():
    if "user_id" not in session or session.get("is_admin"):
        return redirect(url_for("login"))
    user = User.query.get(session["user_id"])
    predictions = Prediction.query.filter_by(user_id=user.id).all()
    return render_template("user_dashboard.html", user=user, predictions=predictions)

@app.route("/admin_dashboard")
def admin_dashboard():
    if "user_id" not in session or not session.get("is_admin"):
        return redirect(url_for("login"))
    all_preds = Prediction.query.all()
    admin_user = User.query.get(session["user_id"])
    return render_template("admin_dashboard.html", predictions=all_preds, user=admin_user)

# ===============================
# PREDICTION ROUTE
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session:
        return jsonify({"error": "Login required"}), 403
    try:
        file = request.files["image"]
        filename = str(uuid.uuid4()) + "_" + file.filename
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file.save(save_path)

        image_path = "uploads/" + filename  # relative path for static

        img = Image.open(save_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        # ------------------
        # Leaf detection
        # ------------------
        with torch.no_grad():
            out_leaf = leaf_model(img_tensor)
            probs = F.softmax(out_leaf, dim=1)[0]
            max_prob, pred_idx = torch.max(probs, 0)
        leaf_pred = LEAF_CLASS_NAMES[pred_idx.item()]

        if leaf_pred == "Not Leaf" and max_prob.item() > THRESHOLD_LEAF:
            prediction = Prediction(
                image_path=image_path,
                leaf="Not Leaf",
                health="",
                disease="",
                confidence=max_prob.item(),
                user_id=session["user_id"]
            )
            db.session.add(prediction)
            db.session.commit()
            return jsonify({"leaf": "Not Leaf", "image_url": url_for("static", filename=image_path)})

        # ------------------
        # Health detection
        # ------------------
        with torch.no_grad():
            out_health = disease_model(img_tensor)
            probs2 = F.softmax(out_health, dim=1)[0]
            max_prob2, pred_idx2 = torch.max(probs2, 0)
        health_pred = DISEASE_CLASS_NAMES[pred_idx2.item()]

        # ------------------
        # Specific disease
        # ------------------
        specific = ""
        if health_pred == "Unhealthy" and max_prob2.item() > THRESHOLD_UNHEALTHY:
            with torch.no_grad():
                out_spec = specific_disease_model(img_tensor)
                probs3 = F.softmax(out_spec, dim=1)[0]
                top_prob, top_idx = torch.max(probs3, 0)
                specific = SPECIFIC_DISEASE_CLASSES[top_idx.item()]

        prediction = Prediction(
            image_path=image_path,
            leaf="Leaf",
            health=health_pred,
            disease=specific,
            confidence=max_prob2.item(),
            user_id=session["user_id"]
        )
        db.session.add(prediction)
        db.session.commit()

        return jsonify({
            "leaf": "Leaf",
            "health": health_pred,
            "disease": specific,
            "confidence": round(max_prob2.item()*100, 2),
            "image_url": url_for("static", filename=image_path)
        })
    except Exception as e:
        print("Prediction Error:", e)
        return jsonify({"error": "Server error"}), 500

# ===============================
# DELETE PREDICTION
# ===============================
@app.route("/delete_prediction/<int:pred_id>", methods=["POST"])
def delete_prediction(pred_id):
    if "user_id" not in session or not session.get("is_admin"):
        return jsonify({"error": "Unauthorized"}), 403

    prediction = Prediction.query.get(pred_id)
    if not prediction:
        return jsonify({"error": "Prediction not found"}), 404

    # Delete image file
    try:
        file_path = os.path.join(app.static_folder, prediction.image_path)
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print("Error deleting file:", e)

    db.session.delete(prediction)
    db.session.commit()
    return jsonify({"success": True})

# ===============================
# ADMIN CREATION
# ===============================
def create_admins():
    with app.app_context():
        db.create_all()
        admin_emails = [
            "232008812@eastdelta.edu.bd",
            "232008012@eastdelta.edu.bd",
            "232006612@eastdelta.edu.bd",
            "232007712@eastdelta.edu.bd"
        ]
        for email in admin_emails:
            if not User.query.filter_by(email=email).first():
                db.session.add(User(email=email, password=generate_password_hash("111111"), is_admin=True))
        db.session.commit()

# ===============================
# RUN APP
# ===============================
if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    create_admins()
    app.run(host="0.0.0.0", port=5000, debug=True)
