from typing import TypeIs

type Result[T, E: Exception] = T | E


def is_err(value: object) -> TypeIs[Exception]:
    return isinstance(value, Exception)
