# Face Recognition Blurring System

A full-stack application for detecting and blurring faces in videos using RetinaFace and face recognition algorithms.

## Project Structure
```
├── backend/
│   ├── app.py            - Flask API endpoints
├── frontend/
│   ├── public/           - Static assets
│   ├── src/              - React components
│   ├── package.json      - Frontend dependencies
├── .gitignore            - Version control exclusions
```

## Setup
1. **Backend**:
```bash
cd backend
pip install -r requirements.txt
flask run
```

2. **Frontend**:
```bash
cd frontend
npm install
npm start
```

## Features
- Video upload handling
- Face detection with RetinaFace
- Face recognition matching
- Selective face blurring
- Progress tracking API