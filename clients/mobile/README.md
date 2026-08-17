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
| Android 13+, model missing | Ana offers to install it; recording works meanwhile |
| Android 13+, recogniser refuses to start | The session continues as a plain recording; nothing spoken is lost |
| Android 12 and below | Recording only — the OS cannot persist audio from the recogniser |

## Capability detection on Android

Android has two recognisers and they answer different questions. Because this
app sets `requiresOnDeviceRecognition`, the library builds its session with
`createOnDeviceSpeechRecognizer()`, which does not go through the device's
default recognition service at all. So `speech.js` deliberately does **not**
gate on `isRecognitionAvailable()` on Android: a phone can answer "no default
recogniser" and still transcribe Turkish offline perfectly well.

For the same reason `getSupportedLocales()` is asked with no service package
first. Naming `com.google.android.as` sends the call down a package-resolution
path that throws outright on devices where the on-device recogniser is not
published under that name, and the old code read that throw as "Turkish is not
installed" — which silently cost the transcript on every such device. The
package query is now only a fallback.

When the probe cannot answer either way, the app tries to transcribe anyway.
If the recogniser then fails, the `end` handler drops the same session to a
plain `expo-audio` recording rather than leaving the user with a button that
did nothing. Guessing wrong that way costs a few seconds; guessing wrong the
other way costs every transcript on the device.

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
real build.

To put the app on your own phone, build the `preview` profile — a standalone
release APK you install and use without a computer attached:

```bash
npx eas-cli login
npx eas-cli init                                    # once, links the project
npx eas-cli build --profile preview --platform android
```

EAS prints a URL and a QR code when it finishes; open it on the phone and
install the `.apk`. Signing keys are generated and held by EAS.

For day-to-day development, build the `development` profile instead and run
Metro against it:

```bash
npx eas-cli build --profile development --platform android
npm install
npm start
```

Building locally instead of on EAS needs the Android SDK (and JDK 17 or 21 —
newer JDKs are not yet supported by the Android Gradle Plugin):

```bash
npx expo run:android
```

`android/` and `ios/` are generated, not committed. `expo prebuild` writes
them from `app.json`; both `expo run:` and EAS do it for you. If you run
prebuild by hand, note that it also rewrites the `android`/`ios` scripts in
`package.json` to the bare-workflow versions — put those back.

## Before publishing

`ios.bundleIdentifier` and `android.package` in `app.json` are currently
`com.oktayelb.hatirlaf`. Change them if you are publishing under a different
account. The icons in `assets/` are generated placeholders.

## Relationship to the server

The Django app in `server/` is no longer part of this client. It remains the
desktop/research side of the project, where the NLP and LLM pipeline runs.
