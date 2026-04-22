import os
from dotenv import load_dotenv

# Load key.env first (local dev), fall back to .env, then plain env vars (Railway/production)
for _f in ("key.env", ".env"):
    if os.path.exists(_f):
        load_dotenv(_f)
        break

API_KEY    = os.getenv("OKX_API_KEY", "")
SECRET_KEY = os.getenv("OKX_SECRET_KEY", "")
PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")

# "1" = paper trading (demo), "0" = live trading
FLAG = os.getenv("OKX_FLAG", "1")

if not all([API_KEY, SECRET_KEY, PASSPHRASE]):
    raise EnvironmentError("Missing OKX credentials. Fill in key.env (local) or set env vars on Railway.")
