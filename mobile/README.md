# Hatırlaf Mobile

Expo/React Native client for the local Hatırlaf Django API.

## What it includes

- Native audio recording with `expo-audio`
- Offline upload queue for text and audio entries
- Retry sync on app foreground and registered background task
- SecureStore-backed API base URL and session cookie storage
- Calendar agenda and monthly recap views
- Settings screen for server URL, app password, and manual sync
- Local notifications scheduled for future calendar events

## Run

```bash
cd mobile
npm install
npm run start
```

For a physical phone, set the API server in **Ayarlar** to your computer LAN address:

```text
http://YOUR_COMPUTER_IP:8001/api
```

`127.0.0.1` only works inside the simulator/emulator or on the same machine.

## Backend

Run Django with:

```bash
cd backend
HATIRLAF_PRELOAD_MODELS=0 ../.venv/bin/python manage.py runserver 0.0.0.0:8001 --noreload
```

Using `0.0.0.0` lets devices on the same network reach the development server.
