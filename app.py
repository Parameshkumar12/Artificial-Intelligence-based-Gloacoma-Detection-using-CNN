import os
import json
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from datetime import datetime
import secrets

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
os.makedirs(app.instance_path, exist_ok=True)
db_path = os.path.join(app.instance_path, 'users.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{db_path}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    result = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Load model
model = None
class_indices = None

def load_model_and_classes():
    global model, class_indices
    try:
        model = tf.keras.models.load_model('models/final_glaucoma_model.h5')
        with open('models/class_indices.json', 'r') as f:
            class_indices = json.load(f)
        class_indices = {v: k for k, v in class_indices.items()}
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

load_model_and_classes()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if user exists
        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
        
        # Create new user
        new_user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get user's predictions
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    
    normalized_results = [normalize_result(p.result) for p in predictions]

    # Calculate statistics
    total_predictions = len(predictions)
    glaucoma_count = sum(1 for result in normalized_results if result == 'Glaucoma')
    normal_count = total_predictions - glaucoma_count

    chart_dates = [p.timestamp.strftime('%Y-%m-%d') for p in predictions]
    chart_results = normalized_results
    
    return render_template('dashboard.html', 
                         predictions=predictions,
                         total_predictions=total_predictions,
                         glaucoma_count=glaucoma_count,
                         normal_count=normal_count,
                         chart_dates=chart_dates,
                         chart_results=chart_results)

@app.route('/detect', methods=['GET', 'POST'])
@login_required
def detect():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Save file
            filename = secure_filename(f"{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process image and predict
            result, confidence = predict_image(filepath)
            result = normalize_result(result)
            
            # Save prediction to database
            prediction = Prediction(
                user_id=current_user.id,
                image_path=filename,
                result=result,
                confidence=confidence
            )
            db.session.add(prediction)
            db.session.commit()
            
            # Store result in session for display
            session['prediction_result'] = {
                'result': result,
                'confidence': confidence,
                'image_path': filename,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return redirect(url_for('result'))
    
    return render_template('detect.html')

@app.route('/result')
@login_required
def result():
    prediction = session.get('prediction_result', None)
    if not prediction:
        return redirect(url_for('detect'))
    prediction['result'] = normalize_result(prediction.get('result'))
    return render_template('result.html', prediction=prediction)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/profile')
@login_required
def profile():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    total_predictions = len(predictions)
    return render_template('profile.html', predictions=predictions, total_predictions=total_predictions)

@app.route('/api/stats')
@login_required
def api_stats():
    predictions = Prediction.query.filter_by(user_id=current_user.id).all()
    
    # Prepare data for charts
    dates = []
    results = []
    for pred in predictions:
        dates.append(pred.timestamp.strftime('%Y-%m-%d'))
        results.append(normalize_result(pred.result))
    
    return jsonify({
        'dates': dates,
        'results': results,
        'total': len(predictions)
    })

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def normalize_result(value):
    if not value:
        return 'Unknown'
    normalized = value.strip().lower()
    if normalized == 'glaucoma':
        return 'Glaucoma'
    if normalized == 'normal':
        return 'Normal'
    return value

def predict_image(image_path):
    if model is None:
        return "Error", 0.0
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class]) * 100
    
    result = class_indices[predicted_class]

    return normalize_result(result), confidence

@app.route('/api/delete_account', methods=['DELETE'])
@login_required
def delete_account():
    """Delete user account and all associated data"""
    try:
        # Delete all predictions
        Prediction.query.filter_by(user_id=current_user.id).delete()
        
        # Delete user
        db.session.delete(current_user)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Account deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/prediction/<int:prediction_id>', methods=['DELETE'])
@login_required
def delete_prediction(prediction_id):
    """Delete a specific prediction"""
    prediction = Prediction.query.get_or_404(prediction_id)
    
    # Check if prediction belongs to current user
    if prediction.user_id != current_user.id:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        # Delete image file
        if os.path.exists(prediction.image_path):
            os.remove(prediction.image_path)
        
        # Delete database record
        db.session.delete(prediction)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Prediction deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/prediction/<int:prediction_id>', methods=['GET'])
@login_required
def get_prediction_details(prediction_id):
    """Get details of a specific prediction"""
    prediction = Prediction.query.get_or_404(prediction_id)
    
    # Check if prediction belongs to current user
    if prediction.user_id != current_user.id:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    return jsonify({
        'id': prediction.id,
        'result': prediction.result,
        'confidence': prediction.confidence,
        'timestamp': prediction.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'image_path': prediction.image_path
    })

@app.route('/api/stats/detailed')
@login_required
def detailed_stats():
    """Get detailed statistics for charts"""
    predictions = Prediction.query.filter_by(user_id=current_user.id).all()
    
    # Monthly statistics
    monthly_stats = {}
    for pred in predictions:
        month = pred.timestamp.strftime('%Y-%m')
        if month not in monthly_stats:
            monthly_stats[month] = {'glaucoma': 0, 'normal': 0, 'total': 0}
        
        monthly_stats[month]['total'] += 1
        if pred.result == 'Glaucoma':
            monthly_stats[month]['glaucoma'] += 1
        else:
            monthly_stats[month]['normal'] += 1
    
    # Format for charts
    months = sorted(monthly_stats.keys())
    glaucoma_data = [monthly_stats[m]['glaucoma'] for m in months]
    normal_data = [monthly_stats[m]['normal'] for m in months]
    
    # Calculate averages
    avg_confidence = sum(p.confidence for p in predictions) / len(predictions) if predictions else 0
    
    return jsonify({
        'months': months,
        'glaucoma_data': glaucoma_data,
        'normal_data': normal_data,
        'total_predictions': len(predictions),
        'avg_confidence': avg_confidence,
        'glaucoma_count': sum(1 for p in predictions if p.result == 'Glaucoma'),
        'normal_count': sum(1 for p in predictions if p.result == 'Normal')
    })

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    """Update user profile information"""
    # Get form data
    first_name = request.form.get('first_name')
    last_name = request.form.get('last_name')
    phone = request.form.get('phone')
    # Add more fields as needed
    
    # Update user profile (you may need to add these fields to User model)
    flash('Profile updated successfully!', 'success')
    return redirect(url_for('profile'))

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

@app.route('/history')
@login_required
def history():
    """View prediction history with pagination and filtering"""
    page = request.args.get('page', 1, type=int)
    per_page = 12  # Items per page
    
    # Get user's predictions
    predictions_query = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc())
    
    # Pagination
    pagination = predictions_query.paginate(page=page, per_page=per_page, error_out=False)
    predictions = pagination.items
    
    # Calculate statistics
    total_predictions = predictions_query.count()
    glaucoma_count = predictions_query.filter_by(result='Glaucoma').count()
    normal_count = total_predictions - glaucoma_count
    
    # Calculate average confidence
    if total_predictions > 0:
        avg_confidence = sum(p.confidence for p in predictions_query.all()) / total_predictions
    else:
        avg_confidence = 0
    
    return render_template('history.html',
                         predictions=predictions,
                         total_predictions=total_predictions,
                         glaucoma_count=glaucoma_count,
                         normal_count=normal_count,
                         avg_confidence=avg_confidence,
                         current_page=page,
                         total_pages=pagination.pages)

@app.route('/api/predictions/clear_all', methods=['DELETE'])
@login_required
def clear_all_predictions():
    """Delete all predictions for the current user"""
    try:
        # Get all predictions
        predictions = Prediction.query.filter_by(user_id=current_user.id).all()
        
        # Delete image files
        for prediction in predictions:
            if os.path.exists(prediction.image_path):
                os.remove(prediction.image_path)
        
        # Delete database records
        Prediction.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'All predictions deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predictions/export')
@login_required
def export_predictions():
    """Export predictions as CSV"""
    import csv
    from io import StringIO
    
    # Get predictions
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    
    # Create CSV
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['ID', 'Date', 'Time', 'Result', 'Confidence (%)', 'Image Path'])
    
    for pred in predictions:
        cw.writerow([
            pred.id,
            pred.timestamp.strftime('%Y-%m-%d'),
            pred.timestamp.strftime('%H:%M:%S'),
            pred.result,
            f"{pred.confidence:.2f}",
            pred.image_path
        ])
    
    # Create response
    output = si.getvalue()
    si.close()
    
    response = make_response(output)
    response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
    response.headers["Content-type"] = "text/csv"
    
    return response

@app.route('/result/<int:prediction_id>')
@login_required
def view_result(prediction_id):
    """View a specific prediction result"""
    prediction = Prediction.query.get_or_404(prediction_id)
    
    # Check if prediction belongs to current user
    if prediction.user_id != current_user.id:
        abort(403)
    
    return render_template('result.html', prediction={
        'result': prediction.result,
        'confidence': prediction.confidence,
        'image_path': prediction.image_path,
        'timestamp': prediction.timestamp.strftime('%Y-%m-%d %H:%M:%S')
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)