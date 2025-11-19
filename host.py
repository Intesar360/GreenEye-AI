import os
import uuid
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import tempfile

# ===============================
# LOGGING
# ===============================
logging.basicConfig(filename="error.log", level=logging.ERROR)

# ===============================
# CONFIG
# ===============================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Ensure writable directories
try:
    UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
except:
    UPLOAD_DIR = tempfile.mkdtemp()

try:
    INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
    os.makedirs(INSTANCE_DIR, exist_ok=True)
except:
    INSTANCE_DIR = tempfile.mkdtemp()

LEAF_MODEL_PATH = os.path.join(MODEL_DIR, "leaf_detector.pth")
DISEASE_MODEL_PATH = os.path.join(MODEL_DIR, "best_cpu_model.pth")
SPECIFIC_DISEASE_MODEL_PATH = os.path.join(MODEL_DIR, "disease_stage2_best_model.pth")

DEVICE = torch.device("cpu")
LEAF_CLASS_NAMES = ["Leaf", "Not Leaf"]
DISEASE_CLASS_NAMES = ["Dry", "Healthy", "Unhealthy"]
THRESHOLD_LEAF = 0.8
THRESHOLD_UNHEALTHY = 0.6

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

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ===============================
# APP SETUP
# ===============================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "super_secret_123")
DB_PATH = os.path.join(INSTANCE_DIR, "database.db")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
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
# LAZY-LOAD MODELS
# ===============================
leaf_model = None
disease_model = None
specific_disease_model = None

def get_leaf_model():
    global leaf_model
    if leaf_model is None:
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(LEAF_CLASS_NAMES))
        model.load_state_dict(torch.load(LEAF_MODEL_PATH, map_location=DEVICE))
        model.eval()
        leaf_model = model
    return leaf_model

def get_disease_model():
    global disease_model
    if disease_model is None:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, len(DISEASE_CLASS_NAMES)))
        model.load_state_dict(torch.load(DISEASE_MODEL_PATH, map_location=DEVICE))
        model.eval()
        disease_model = model
    return disease_model

def get_specific_disease_model():
    global specific_disease_model
    if specific_disease_model is None:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, len(SPECIFIC_DISEASE_CLASSES)))
        model.load_state_dict(torch.load(SPECIFIC_DISEASE_MODEL_PATH, map_location=DEVICE))
        model.eval()
        specific_disease_model = model
    return specific_disease_model

# ===============================
# ROUTES
# ===============================
@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("admin_dashboard") if session.get("is_admin") else url_for("user_dashboard"))
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
    
    # CHANGE IS HERE: Added .order_by(Prediction.timestamp.desc())
    predictions = Prediction.query.filter_by(user_id=user.id).order_by(Prediction.timestamp.desc()).all()
    
    return render_template("user_dashboard.html", user=user, predictions=predictions)

@app.route("/admin_dashboard")
def admin_dashboard():
    if "user_id" not in session or not session.get("is_admin"):
        return redirect(url_for("login"))
    
    # CHANGE IS HERE: Added .order_by(Prediction.timestamp.desc())
    all_preds = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    
    admin_user = User.query.get(session["user_id"])
    return render_template("admin_dashboard.html", predictions=all_preds, user=admin_user)

# ===============================
# PREDICTION ROUTE
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    import traceback
    if "user_id" not in session:
        return jsonify({"error": "Login required"}), 403

    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Save file
        filename = str(uuid.uuid4()) + "_" + file.filename
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file.save(save_path)
        image_path = "uploads/" + filename

        # Load image
        img = Image.open(save_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        # Leaf detection
        leaf_model = get_leaf_model()
        with torch.no_grad():
            out_leaf = leaf_model(img_tensor)
            probs = F.softmax(out_leaf, dim=1)[0]
            max_prob, pred_idx = torch.max(probs, 0)
        leaf_pred = LEAF_CLASS_NAMES[pred_idx.item()]

        if leaf_pred == "Not Leaf" and max_prob.item() > THRESHOLD_LEAF:
            pred = Prediction(
                image_path=image_path,
                leaf="Not Leaf",
                health="",
                disease="",
                confidence=max_prob.item(),
                user_id=session["user_id"]
            )
            db.session.add(pred)
            db.session.commit()
            return jsonify({"leaf": "Not Leaf", "image_url": url_for("static", filename=image_path)})

        # Health detection
        disease_model = get_disease_model()
        with torch.no_grad():
            out_health = disease_model(img_tensor)
            probs2 = F.softmax(out_health, dim=1)[0]
            max_prob2, pred_idx2 = torch.max(probs2, 0)
        health_pred = DISEASE_CLASS_NAMES[pred_idx2.item()]

        # Specific disease detection
        specific = ""
        if health_pred == "Unhealthy" and max_prob2.item() > THRESHOLD_UNHEALTHY:
            specific_disease_model = get_specific_disease_model()
            with torch.no_grad():
                out_spec = specific_disease_model(img_tensor)
                probs3 = F.softmax(out_spec, dim=1)[0]
                top_prob, top_idx = torch.max(probs3, 0)
                specific = SPECIFIC_DISEASE_CLASSES[top_idx.item()]

        # Save prediction
        pred = Prediction(
            image_path=image_path,
            leaf="Leaf",
            health=health_pred,
            disease=specific,
            confidence=max_prob2.item(),
            user_id=session["user_id"]
        )
        db.session.add(pred)
        db.session.commit()

        return jsonify({
            "leaf": "Leaf",
            "health": health_pred,
            "disease": specific,
            "confidence": round(max_prob2.item()*100, 2),
            "image_url": url_for("static", filename=image_path)
        })

    except Exception as e:
        logging.error("Prediction error", exc_info=True)
        traceback.print_exc()
        return jsonify({"error": "Server error", "details": str(e)}), 500

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

    try:
        file_path = os.path.join(app.static_folder, prediction.image_path)
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logging.error("Error deleting file", exc_info=True)

    db.session.delete(prediction)
    db.session.commit()
    return jsonify({"success": True})

# ===============================
# ADMIN USER MANAGEMENT ROUTES
# ===============================
@app.route('/admin/users')
def admin_users():
    if "user_id" not in session or not session.get("is_admin"):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    users = User.query.all()
    user_list = []
    
    for user in users:
        user_data = {
            'id': user.id,
            'email': user.email,
            'first_name': getattr(user, 'first_name', None),
            'last_name': getattr(user, 'last_name', None),
            'created_at': user.created_at.isoformat() if hasattr(user, 'created_at') and user.created_at else None,
            'last_login': user.last_login.isoformat() if hasattr(user, 'last_login') and user.last_login else None,
            'status': getattr(user, 'status', 'active'),
            'role': 'admin' if user.is_admin else 'user',
            'prediction_count': Prediction.query.filter_by(user_id=user.id).count()
        }
        user_list.append(user_data)
    
    return jsonify({'success': True, 'users': user_list})

@app.route('/admin/users/<int:user_id>')
def admin_user_detail(user_id):
    if "user_id" not in session or not session.get("is_admin"):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    user = User.query.get_or_404(user_id)
    
    user_data = {
        'id': user.id,
        'email': user.email,
        'first_name': getattr(user, 'first_name', None),
        'last_name': getattr(user, 'last_name', None),
        'created_at': user.created_at.isoformat() if hasattr(user, 'created_at') and user.created_at else None,
        'last_login': user.last_login.isoformat() if hasattr(user, 'last_login') and user.last_login else None,
        'status': getattr(user, 'status', 'active'),
        'role': 'admin' if user.is_admin else 'user',
        'prediction_count': Prediction.query.filter_by(user_id=user.id).count()
    }
    
    return jsonify({'success': True, 'user': user_data})

@app.route('/admin/users/<int:user_id>', methods=['PUT'])
def admin_update_user(user_id):
    if "user_id" not in session or not session.get("is_admin"):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    user = User.query.get_or_404(user_id)
    data = request.get_json()
    
    if 'status' in data:
        user.status = data['status']
    
    try:
        db.session.commit()
        return jsonify({'success': True, 'message': 'User updated successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/users/<int:user_id>', methods=['DELETE'])
def admin_delete_user(user_id):
    if "user_id" not in session or not session.get("is_admin"):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    user = User.query.get_or_404(user_id)
    
    try:
        # Delete user's predictions first
        Prediction.query.filter_by(user_id=user_id).delete()
        db.session.delete(user)
        db.session.commit()
        return jsonify({'success': True, 'message': 'User deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    
 # ===============================
# ADMIN STATISTICS ROUTE
# ===============================
@app.route('/admin/statistics')
def admin_statistics():
    if "user_id" not in session or not session.get("is_admin"):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        # Get real counts from database
        total_users = User.query.count()
        total_predictions = Prediction.query.count()
        
        # Count predictions by health status
        healthy_plants = Prediction.query.filter_by(health="Healthy").count()
        diseased_plants = Prediction.query.filter_by(health="Unhealthy").count()
        dry_plants = Prediction.query.filter_by(health="Dry").count()
        
        statistics = {
            'total_users': total_users,
            'total_predictions': total_predictions,
            'healthy_plants': healthy_plants,
            'diseased_plants': diseased_plants,
            'dry_plants': dry_plants
        }
        
        return jsonify({'success': True, 'statistics': statistics})
        
    except Exception as e:
        logging.error("Statistics error", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500   
# ===============================
# ADMIN ANALYTICS ROUTES
# ===============================
@app.route('/admin/analytics/health-distribution')
def admin_health_distribution():
    if "user_id" not in session or not session.get("is_admin"):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        total_predictions = Prediction.query.count()
        
        if total_predictions == 0:
            return jsonify({
                'success': True, 
                'distribution': {'healthy': 0, 'diseased': 0, 'dry': 0},
                'percentages': {'healthy': 0, 'diseased': 0, 'dry': 0}
            })
        
        healthy_count = Prediction.query.filter_by(health="Healthy").count()
        diseased_count = Prediction.query.filter_by(health="Unhealthy").count()
        dry_count = Prediction.query.filter_by(health="Dry").count()
        
        healthy_percent = round((healthy_count / total_predictions) * 100)
        diseased_percent = round((diseased_count / total_predictions) * 100)
        dry_percent = round((dry_count / total_predictions) * 100)
        
        return jsonify({
            'success': True,
            'distribution': {
                'healthy': healthy_count,
                'diseased': diseased_count,
                'dry': dry_count
            },
            'percentages': {
                'healthy': healthy_percent,
                'diseased': diseased_percent,
                'dry': dry_percent
            }
        })
        
    except Exception as e:
        logging.error("Health distribution error", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/analytics/user-stats')
def admin_user_stats():
    if "user_id" not in session or not session.get("is_admin"):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        users = User.query.all()
        user_stats = []
        
        for user in users:
            predictions = Prediction.query.filter_by(user_id=user.id).all()
            total_predictions = len(predictions)
            
            healthy_count = sum(1 for p in predictions if p.health == "Healthy")
            diseased_count = sum(1 for p in predictions if p.health == "Unhealthy")
            dry_count = sum(1 for p in predictions if p.health == "Dry")
            
            user_stats.append({
                'email': user.email,
                'total_predictions': total_predictions,
                'healthy': healthy_count,
                'diseased': diseased_count,
                'dry': dry_count
            })
        
        # Sort by total predictions (descending)
        user_stats.sort(key=lambda x: x['total_predictions'], reverse=True)
        
        return jsonify({'success': True, 'user_stats': user_stats})
        
    except Exception as e:
        logging.error("User stats error", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/analytics/top-diseases')
def admin_top_diseases():
    if "user_id" not in session or not session.get("is_admin"):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        # Get all predictions with diseases
        predictions_with_disease = Prediction.query.filter(
            Prediction.disease != '', 
            Prediction.disease.isnot(None)
        ).all()
        
        total_disease_predictions = len(predictions_with_disease)
        
        if total_disease_predictions == 0:
            return jsonify({'success': True, 'top_diseases': []})
        
        # Count diseases
        disease_counts = {}
        for prediction in predictions_with_disease:
            disease = prediction.disease
            if disease:
                disease_counts[disease] = disease_counts.get(disease, 0) + 1
        
        # Convert to list and calculate percentages
        top_diseases = []
        for disease, count in disease_counts.items():
            percentage = round((count / total_disease_predictions) * 100)
            top_diseases.append({
                'disease': disease,
                'count': count,
                'percentage': percentage
            })
        
        # Sort by count (descending) and take top 10
        top_diseases.sort(key=lambda x: x['count'], reverse=True)
        top_diseases = top_diseases[:10]
        
        return jsonify({'success': True, 'top_diseases': top_diseases})
        
    except Exception as e:
        logging.error("Top diseases error", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

# ===============================
# USER STATISTICS ROUTE
# ===============================
@app.route('/user/statistics')
def user_statistics():
    if "user_id" not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        user_id = session["user_id"]
        
        # Get counts for the current user only
        total_predictions = Prediction.query.filter_by(user_id=user_id).count()
        healthy_plants = Prediction.query.filter_by(user_id=user_id, health="Healthy").count()
        diseased_plants = Prediction.query.filter_by(user_id=user_id, health="Unhealthy").count()
        dry_plants = Prediction.query.filter_by(user_id=user_id, health="Dry").count()
        
        statistics = {
            'total_predictions': total_predictions,
            'healthy_plants': healthy_plants,
            'diseased_plants': diseased_plants,
            'dry_plants': dry_plants
        }
        
        return jsonify({'success': True, 'statistics': statistics})
        
    except Exception as e:
        logging.error("User statistics error", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500   
    
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
# =====================================================
# NEW: IOT AUTOMATION ROUTE 
# =====================================================
@app.route("/api/iot/upload", methods=["POST"])
def iot_upload():
    API_KEY = "greeneye_secret_pass_123"
    if request.headers.get("X-API-KEY") != API_KEY:
        return jsonify({"error": "Invalid API Key"}), 401

    try:
        # --- FIX: GET EMAIL FROM SCRIPT ---
        target_email = request.form.get("email")
        
        # Fallback if script didn't send email (prevents crash)
        if not target_email: 
             return jsonify({"error": "Email missing"}), 400

        user = User.query.filter_by(email=target_email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404
        # ----------------------------------

        if "image" not in request.files:
            return jsonify({"error": "No image part"}), 400
        
        file = request.files["image"]
        filename = "iot_" + str(uuid.uuid4()) + ".jpg"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file.save(save_path)
        image_path = "uploads/" + filename

        img = Image.open(save_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        # Leaf Model
        leaf_model = get_leaf_model()
        with torch.no_grad():
            out_leaf = leaf_model(img_tensor)
            probs = F.softmax(out_leaf, dim=1)[0]
            max_prob, pred_idx = torch.max(probs, 0)
        leaf_pred = LEAF_CLASS_NAMES[pred_idx.item()]

        # Not Leaf Handling
        if leaf_pred == "Not Leaf" and max_prob.item() > THRESHOLD_LEAF:
            pred = Prediction(
                image_path=image_path, leaf="Not Leaf", health="", disease="",
                confidence=max_prob.item(), user_id=user.id
            )
            db.session.add(pred)
            db.session.commit()
            # Return 'user' so script doesn't say 'None'
            return jsonify({"status": "success", "result": "Not Leaf", "user": user.email})

        # Disease Models
        disease_model = get_disease_model()
        with torch.no_grad():
            out_health = disease_model(img_tensor)
            probs2 = F.softmax(out_health, dim=1)[0]
            max_prob2, pred_idx2 = torch.max(probs2, 0)
        health_pred = DISEASE_CLASS_NAMES[pred_idx2.item()]

        specific = ""
        if health_pred == "Unhealthy" and max_prob2.item() > THRESHOLD_UNHEALTHY:
            specific_disease_model = get_specific_disease_model()
            with torch.no_grad():
                out_spec = specific_disease_model(img_tensor)
                probs3 = F.softmax(out_spec, dim=1)[0]
                top_prob, top_idx = torch.max(probs3, 0)
                specific = SPECIFIC_DISEASE_CLASSES[top_idx.item()]

        # Save to DB
        pred = Prediction(
            image_path=image_path, leaf="Leaf", health=health_pred,
            disease=specific, confidence=max_prob2.item(), user_id=user.id
        )
        db.session.add(pred)
        db.session.commit()

        return jsonify({
            "status": "success", "health": health_pred, 
            "disease": specific, "user": user.email
        })

    except Exception as e:
        logging.error("IoT Upload Error", exc_info=True)
        return jsonify({"error": str(e)}), 500
# ===============================
# RUN APP
# ===============================
if __name__ == "__main__":
    create_admins()
    app.run(host="0.0.0.0", port=5000, debug=False)
