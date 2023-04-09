from importlib.metadata import version as importlib_version
from autoeval.pipeline import (
    AgreementFilter,
    PersonaStatement,
    PersonaStatementChain,
    AgreementFilterChain,
)

__version__ = importlib_version("autoeval")

__all__ = [
    "AgreementFilter",
    "PersonaStatement",
    "PersonaStatementChain",
    "AgreementFilterChain",
]
