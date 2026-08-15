# Hatırlaf Mobile

Expo/React Native client. **Everything stays on the phone** — there is no
server, no account, and no network call anywhere in this client.

## What it includes

- Turkish speech-to-text that runs on the device, never over the network
- Audio recording saved to the phone's own storage
- A local SQLite diary, with playback and editable text
- App password held in the iOS keychain / Android keystore
- Export to a text file via the system share sheet

## The privacy rule

`src/services/speech.js` always sets `requiresOnDeviceRecognition: true`. Both
platforms will silently fall back to network recognition otherwise, which on
Android means sending diary audio to Google. When on-device Turkish is not
available the app records audio without a transcript rather than going online.

That has consequences worth knowing:

| Situation | What happens |
|---|---|
| iOS, modern device | Live Turkish transcript plus the recording |
| Android 13+, Turkish model installed | Live Turkish transcript plus the recording |
| Android 13+, model missing | Ayarlar offers to install it; recording works meanwhile |
| Android 12 and below | Recording only — the OS cannot persist audio from the recogniser |

## Storage

Recordings are 16 kHz 16-bit mono WAV, roughly **1.9 MB per minute**. Ayarlar
shows the running total. The format is fixed by the speech recogniser: it will
not accept the compressed AAC that `expo-audio` produces, so there is no
"record compressed, transcribe later" path without a transcoder.

Because the phone is the only copy, **Ayarlar → Günlüğümü Dışa Aktar** writes
every entry's text to one file and hands it to the share sheet. Audio is shared
one entry at a time from Günlüğüm; there is no bulk audio archive yet.

## Run

This client uses native modules, so **Expo Go will not run it**. You need a
development build.

```bash
# once, to get a build onto a device
npx eas-cli build --profile development --platform android

# then, day to day
npm install
npm start
```

Building locally instead of on EAS needs the Android SDK (and JDK 17 or 21 —
newer JDKs are not yet supported by the Android Gradle Plugin):

```bash
npx expo run:android
```

## Before publishing

`ios.bundleIdentifier` and `android.package` in `app.json` are currently
`com.oktayelb.hatirlaf`. Change them if you are publishing under a different
account. The icons in `assets/` are generated placeholders.

## Relationship to the server

The Django app in `server/` is no longer part of this client. It remains the
desktop/research side of the project, where the NLP and LLM pipeline runs.
