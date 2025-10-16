# MOF Optimization SaaS Platform

A SaaS platform for **Metal-Organic Frameworks (MOFs)** and other porous materials (e.g., zeolites), designed to optimize and predict material properties for **carbon capture applications**. The platform leverages AI/ML models, RAG (Retrieval-Augmented Generation), and advanced visualization to assist researchers and corporate clients in material discovery, property prediction, and synthesis optimization.

## Features

### Material Search & Recommendation

* Search MOFs by chemical and structural properties.
* Retrieve high-surface-area MOFs or other criteria.
* LLM-powered recommendations for similar materials and optimal synthesis conditions.
* Filter materials by adsorption efficiency and performance metrics.

### Property Prediction

* Upload CIF files to predict MOF properties using AI models.
* Generate CO₂ adsorption curves and visualize data.
* Support for image-based analysis of adsorption data.

### RAG & Knowledge Retrieval

* Ask detailed questions about MOFs or related materials.
* Receive context-aware explanations from scientific literature and internal datasets.

### User Experience & Dashboard

* Interactive material search and recommendation UI.
* Visualization dashboard with sortable tables, graphs, and charts.
* Export results as PDF, HTML, JSON, or visual charts (PNG/HTML).
* User authentication, activity tracking, and favorites management.

### SaaS Platform Security

* Secure API with HTTPS, authentication, and encrypted data storage.
* Firebase-based activity and prediction logging.
* Scalable backend using FastAPI and PostgreSQL.

## Tech Stack

* **Backend:** Python, FastAPI, SQLAlchemy
* **Frontend:** React, Plotly
* **Database:** PostgreSQL, FAISS (vector DB)
* **AI/ML:** LangChain, Google Vertex AI Gemini, CIF-based property prediction models
* **Authentication & Logging:** Firebase Admin SDK
* **Visualization:** Plotly, PNG/HTML graph generation

## Installation & Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/mof-optimization-saas.git
   cd mof-optimization-saas
   ```
2. Set up environment variables and Firebase credentials.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Run the FastAPI server:

   ```bash
   uvicorn main:app --reload
   ```
5. Access the API at `http://localhost:8000` or connect the React frontend for full SaaS experience.

## Contribution

Contributions are welcome! Please create an issue or pull request for bug fixes, new features, or improvements.

## License

[MIT License](LICENSE)


