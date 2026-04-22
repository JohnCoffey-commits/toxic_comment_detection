"""Configuration objects for the preprocessing pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for deterministic preprocessing."""

    project_root: Path = Path(__file__).resolve().parents[2]
    jigsaw_raw_path: Path = Path("dataset/jigsaw/raw/train.csv")
    civil_raw_path: Path = Path("dataset/civil_comments/raw/train.csv")
    processed_dir: Path = Path("data/processed")
    metadata_dir: Path = Path("metadata")
    report_dir: Path = Path("reports/preprocessing")
    civil_threshold: float = 0.5
    civil_identity_threshold: float = 0.5
    random_seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.15
    civil_external_test_size: float = 0.20
    url_replacement: str = "<URL>"
    email_replacement: str = "<EMAIL>"
    replace_urls: bool = True
    replace_emails: bool = True
    duplicate_lowercase: bool = True
    remove_missing_text: bool = True
    remove_missing_labels: bool = True
    remove_normalized_duplicates: bool = True
    tokenizer_name: str = "distilbert-base-uncased"
    tokenizer_batch_size: int = 1024
    require_transformers_tokenizer: bool = True
    fail_on_leakage: bool = True
    label_drift_warning_threshold: float = 0.03
    suspicious_split_min_rows: int = 100
    identity_terms: tuple[str, ...] = field(
        default_factory=lambda: (
            "asian",
            "atheist",
            "bisexual",
            "black",
            "buddhist",
            "christian",
            "disabled",
            "disability",
            "female",
            "gay",
            "heterosexual",
            "hindu",
            "homosexual",
            "jewish",
            "latino",
            "lesbian",
            "male",
            "muslim",
            "psychiatric",
            "queer",
            "trans",
            "transgender",
            "white",
            "woman",
            "women",
            "man",
            "men",
        )
    )
    explicit_toxic_terms: tuple[str, ...] = field(
        default_factory=lambda: (
            "asshole",
            "bastard",
            "bitch",
            "bullshit",
            "cunt",
            "dick",
            "dumbass",
            "fuck",
            "idiot",
            "moron",
            "shit",
            "slut",
            "whore",
        )
    )

    def resolve(self, path: Path) -> Path:
        """Resolve a project-relative path to an absolute path."""

        return path if path.is_absolute() else self.project_root / path

    @property
    def jigsaw_raw_abs(self) -> Path:
        return self.resolve(self.jigsaw_raw_path)

    @property
    def civil_raw_abs(self) -> Path:
        return self.resolve(self.civil_raw_path)

    @property
    def processed_abs(self) -> Path:
        return self.resolve(self.processed_dir)

    @property
    def metadata_abs(self) -> Path:
        return self.resolve(self.metadata_dir)

    @property
    def report_abs(self) -> Path:
        return self.resolve(self.report_dir)

    def to_serializable_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML friendly representation."""

        out = asdict(self)
        for key, value in list(out.items()):
            if isinstance(value, Path):
                out[key] = str(value)
            elif isinstance(value, tuple):
                out[key] = list(value)
        return out


JIGSAW_TEXT_COLUMN = "comment_text"
JIGSAW_LABEL_COLUMNS = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)

CIVIL_TEXT_CANDIDATES = ("comment_text", "text")
CIVIL_LABEL_CANDIDATES = ("target", "toxicity")
CIVIL_IDENTITY_COLUMNS = (
    "asian",
    "atheist",
    "bisexual",
    "black",
    "buddhist",
    "christian",
    "female",
    "heterosexual",
    "hindu",
    "homosexual_gay_or_lesbian",
    "intellectual_or_learning_disability",
    "jewish",
    "latino",
    "male",
    "muslim",
    "other_disability",
    "other_gender",
    "other_race_or_ethnicity",
    "other_religion",
    "other_sexual_orientation",
    "physical_disability",
    "psychiatric_or_mental_illness",
    "transgender",
    "white",
)

