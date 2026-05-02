import os
from dotenv import load_dotenv

# Try to load credentials from key.env (local dev) first, then .env.
# On Railway (production), neither file exists — credentials come from Railway's env var panel.
for _f in ("key.env", ".env"):
    if os.path.exists(_f):
        load_dotenv(_f)
        break

# Pull each OKX credential from the environment
API_KEY    = os.getenv("OKX_API_KEY", "")
SECRET_KEY = os.getenv("OKX_SECRET_KEY", "")
PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")

# "1" = paper trading (safe demo account), "0" = real live trading with real money
FLAG = os.getenv("OKX_FLAG", "1")

# Crash early with a clear message if any credential is missing
if not all([API_KEY, SECRET_KEY, PASSPHRASE]):
    raise EnvironmentError("Missing OKX credentials. Fill in key.env (local) or set env vars on Railway.")
