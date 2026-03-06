
## 2024-05-20 - [Optimize whitespace sanitization]
**Learning:** `re.sub(r"\s+", " ", text).strip()` is significantly slower than `" ".join(text.split())` for whitespace normalization in Python.
**Action:** Use `" ".join(text.split())` in hot paths like `sanitize_content` where text processing throughput matters.
