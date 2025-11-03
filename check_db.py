from host import db, User, Prediction, app

with app.app_context():
    print("Users:", User.query.all())
    print("Predictions:", Prediction.query.all())
