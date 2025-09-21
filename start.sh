!pip install -r backend/requirements.txt
!cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT
!cd ../frontend && npm start
