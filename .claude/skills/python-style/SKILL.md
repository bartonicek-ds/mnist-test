---
name: python-style
description: Python style guidelines (None checks, comments, comprehensions, class hierarchies, function arguments, globals, errors as values). Use when writing, editing or reviewing Python code.
---

# Python style

These apply to all Python in the project: application code, tests, jobs and scripts.

## No implicit None checks

Always write `if x is None` / `if x is not None`, never `if not x` / `if x`, when the value can be
None. Truthiness also treats `0`, `""`, `[]` and `{}` as missing, which hides bugs when those are
valid.

```python
# Good
if middle_name is None:
    return first_name

if limit is not None:
    rows = rows[:limit]

# Avoid
if not middle_name:
    return first_name

if limit:
    rows = rows[:limit]
```

Don't rely on truthiness for other values either; compare explicitly, e.g. `if len(rows) == 0`
instead of `if not rows`, and `if name == ""` instead of `if not name`.

## Short comments and docstrings

Keep comments and docstrings to 1-2 lines. Say why, not what the code already says. Go longer only
for something important or genuinely hard to explain; detailed reasoning belongs in the commit or
the docs. This also applies to YAML, Makefiles and scripts.

## Prefer postponed annotations over quoting

Prefer `from __future__ import annotations` over quoting types. For a type only imported under
`TYPE_CHECKING`, write `view: DataclassInstance | None`, not `view: "DataclassInstance | None"`.
Quoting is fine where the import doesn't help (`cast("User", value)`, `TypeVar` bounds), or for a
single annotation in a module FastAPI or pydantic read at runtime.

## Readable comprehensions and loops

- Comprehensions and generator expressions should be at most 1-2 lines. Longer ones become a
  well-named intermediate variable, a small helper function, or a plain `for` loop.
- Loop variables get descriptive names (`path`, `row`, `user`), not single letters.

```python
# Good
active_names = [user.name for user in users if user.is_active]

# Avoid
r = [x for x in (p.parent for p in paths if p.suffix == ".txt") if os.access(x, os.R_OK) and x not in seen]
```

## Blank line after blocks

Leave an empty line after an `if`, `match`, `for`, `while`, `with` or `try` block, before the next
statement at the same level, so it's clear where the block ends. A chain like `if`/`elif`/`else`
or `try`/`except` counts as one block. Not needed when the block is the last thing in its enclosing
block.

```python
# Good
if len(rows) == 0:
    return []

users = parse_users(rows)

# Avoid
if len(rows) == 0:
    return []
users = parse_users(rows)
```

## Flat class hierarchies

Prefer some duplication over nested class hierarchies, so a class's behaviour is visible in the
class itself rather than in its ancestors. Subclass an interface directly, without intermediate
base classes, and share code through plain functions or by calling another class's method.

## Pass only what a function needs

Functions should take few, well-named arguments with specific types: the values they use, not a
broad object they dig them out of (a `Request`, a config, a client used for one field). The
signature then shows what the function depends on, and tests can pass plain values. Read broad
objects once at the edge (e.g. a request handler) and pass their parts on. A small, typed group of
values is fine when several functions share them and it clearly helps readability.

```python
# Good
def get_language(accept_language: str | None) -> str:
    return (accept_language or "en").split(",")[0]

language = get_language(request.headers.get("accept-language"))

# Avoid: the signature hides that only one header is used
def get_language(request: Request) -> str:
    return (request.headers.get("accept-language") or "en").split(",")[0]
```

## Pass values in rather than reading globals

Don't read globals (a module-level config, client or cache) deep inside a function: the signature
doesn't show the dependency, and tests have to patch it. Read them at the edge and pass the values
in.

```python
# Good
def is_expired(created_at: datetime, max_age: timedelta, now: datetime) -> bool:
    return now - created_at > max_age

# Avoid: the setting and the clock are hidden dependencies
def is_expired(created_at: datetime) -> bool:
    return datetime.now() - created_at > Settings.MAX_AGE
```

## Result type and errors as values

Expected failures are returned as values, so a function's signature shows whether and how it can
fail, and its caller decides what to do. Only top-level code (a job's entry point, a request
handler) raises, e.g. to fail the run.

- Functions that can fail return `Result[T, E]` (`type Result[T, E: Exception] = T | E`): the value,
  or an exception as the error. Check with `is_err(value)` (a `TypeIs[Exception]`), then use the
  value directly.
- A successful value must never be an exception, since that's how the two are told apart.
- Use distinct error types for failures that need different handling.
- Bugs and broken invariants can still raise (or `assert`) where they happen.
- `Result` and `is_err` live in a small `result.py`; packages deployed separately each keep their
  own copy rather than sharing one.

```python
# Good
def load_settings(path: Path) -> Result[Settings, OSError]:
    try:
        text = path.read_text()
    except OSError as error:
        return error
    ...

settings = load_settings(path)
if is_err(settings):
    record_failure(path, settings)
    continue

# Avoid: raising deep down, and catching broadly far away
def load_settings(path: Path) -> Settings:
    text = path.read_text()  # an OSError escapes to whoever happens to catch it
    ...
```
