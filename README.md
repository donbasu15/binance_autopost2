# Binance Square Auto-Poster Bot

An automated signal-driven Binance Square poster featuring a Flask web server, real-time glassmorphism web dashboard, multi-key rotation, and a thread-safe scheduler.

## Deployment on Render

To deploy this application as a **Web Service** on Render:

### 1. Repository Configuration
- **Runtime**: Python
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn --workers 1 bipi:app`
  *(Note: It is crucial to use `--workers 1` to prevent Gunicorn from launching multiple worker processes, which would cause duplicate scheduler execution and API quota exhaustion.)*

### 2. Environment Variables
Add the following Environment Variables in the Render dashboard:

| Variable | Description |
|---|---|
| `GEMINI_API_KEYS` | A comma-separated list of Gemini API keys (e.g. `key1,key2,key3`) for rotation and failover. |
| `BINANCE_SQUARE_KEY` | Your Binance Square OpenAPI key. |
| `CRYPTOPANIC_KEY` | *(Optional)* Your Cryptopanic API key for filtering hot news. |
| `PORT` | Render automatically binds this; defaults to `10000` or `5000`. |

---

## Local Development

### Installation
1. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up a `.env` file:
   ```ini
   GEMINI_API_KEYS=your_gemini_key1,your_gemini_key2
   BINANCE_SQUARE_KEY=your_binance_square_key
   CRYPTOPANIC_KEY=your_cryptopanic_key
   PORT=5000
   ```

### Run Server + Dashboard
```bash
python bipi.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### Run in CLI Mode (No Web Server)
```bash
python bipi.py --cli
```
