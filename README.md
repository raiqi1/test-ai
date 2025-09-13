# Project Setup

## 1. Clone Repository
```bash
git clone https://github.com/username/repo.git
cd repo
```

## 2. Setup Environment
Copy the `.env.example` file to `.env` and adjust the values as needed.
```bash
cp .env.example .env
```

## 3. Install Dependencies
Make sure **Python 3.9+** is installed, then run:
```bash
pip3 install -r requirements.txt
```

## 4. Run the Project
Start the server with **uvicorn**:
```bash
uvicorn main:app --reload
```

If you want to expose it publicly (e.g., on a server), use:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Notes
- Use `python3 -m venv venv && source venv/bin/activate` to create a virtual environment (optional but recommended).
- Ensure all environment variables in `.env` are set correctly.
