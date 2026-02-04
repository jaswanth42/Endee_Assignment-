# 🛒 QuickCart – Endee AI Assignment

This project is a simple product search and ingestion system built as part of the **Endee AI assignment**.
It demonstrates data ingestion, search functionality, and clean project structuring using Python.



 📁 Project Structure

```
quickcart-endee-ai/
│
├── data/
│   └── products.json        # Sample product data
│
├── src/
│   ├── app.py               # Entry point of the application
│   ├── ingest.py            # Script to ingest product data
│   └── search.py            # Search functionality implementation
│
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignored files and folders
└── README.md                # Project documentation
```

---

⚙️ Setup Instructions

1️⃣ Clone the repository

```bash
git clone https://github.com/jaswanth42/Endee_Assignment-.git
cd quickcart-endee-ai
```

---

 2️⃣ Create and activate virtual environment

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

---

 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

🚀 How to Run the Project

# Ingest product data

```bash
python src/ingest.py
```

# Search products

```bash
python src/search.py
```

*(Modify search queries inside `search.py` if required)*

---

# 🧠 Features

* Product data ingestion from JSON
* Search functionality over ingested data
* Clean and modular Python code
* Virtual environment support
* Proper Git hygiene (`venv` and secrets ignored)

---

# 🛠️ Tech Stack

* **Python**
* **JSON** for data storage
* **Virtualenv** for environment management


# 📌 Notes

* `venv/` and `.env` are intentionally excluded from version control
* This project focuses on logic clarity and structure rather than UI
