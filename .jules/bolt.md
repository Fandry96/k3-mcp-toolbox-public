## 2024-03-24 - [Re.sub whitespace normalization vs string split]
**Learning:** `re.sub(r"\s+", " ", text).strip()` is significantly slower than `" ".join(text.split())` in Python for whitespace normalization.
**Action:** Replace `re.sub(r"\s+", " ", text).strip()` with `" ".join(text.split())` in the `sanitize_content` method of `MatryoshkaIndexer` to improve indexing performance.
