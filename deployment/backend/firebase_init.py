# firebase_init.py
import firebase_admin
from firebase_admin import credentials, firestore

def get_db():
    if not firebase_admin._apps:
        cred = credentials.Certificate("/Users/nemekhbayarnomin/Documents/Intern/mof/mof-application/mof-firebase-b6b3b-firebase-adminsdk-fbsvc-f23d58405d.json")
        firebase_admin.initialize_app(cred)
    return firestore.client()
