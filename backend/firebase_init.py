# firebase_init.py
import firebase_admin
from firebase_admin import credentials, firestore

def init_firebase_app():
    if not firebase_admin._apps:
        cred = credentials.Certificate("/service_account.json")
        firebase_admin.initialize_app(cred)
    return firestore.client()

