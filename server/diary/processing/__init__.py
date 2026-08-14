"""Local, privacy-preserving NLP pipeline.

Components are deliberately decoupled:
  * ``transcription`` — audio -> text (Whisper family)
  * ``nlp``           — text -> lemmas + entity mentions (Zeyrek + NER)
  * ``conflicts``     — mentions -> flagged conflicts with fallback plan
  * ``pipeline``      — end-to-end orchestration triggered after upload

Every component degrades gracefully when optional ML deps are missing so the
rest of the system remains exercisable without GB of downloads.
"""
