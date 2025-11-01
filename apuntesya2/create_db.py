from app import app, Base, engine

if __name__ == "__main__":
    with app.app_context():
        Base.metadata.create_all(engine)
        print("DB creada en", app.config['SQLALCHEMY_DATABASE_URI'])