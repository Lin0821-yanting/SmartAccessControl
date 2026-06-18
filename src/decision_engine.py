#!/usr/bin/env python3
# Copyright (c) 2026 <Your Name>, <Partner's Name>
# Tatung University — I4210 AI實務專題
"""src/decision_engine.py — access-control decision state machine.

Consumes per-frame AI pipeline outputs (similarity, anti_spoof_pass,
face_in_db) and maintains an internal consecutive-frame counter to
produce one of five Decision enum values:

    GRANT   — similarity >= threshold, anti-spoof passed, face in DB,
               consecutive frame count reached required minimum.
    DENY    — face in DB but similarity below threshold.
    UNKNOWN — face detected but not present in the enrolled database.
    SPOOF   — anti-spoof check failed (photo / screen attack detected).
    IGNORE  — no face detected within the gate window; counter reset.

Design constraints
------------------
* Pure-logic class: no GPIO, no MQTT, no hardware dependencies.
  Fully unit-testable on any x86 CI runner without a Jetson.
* Single internal state: ``_consecutive_frames`` (int).
  Reset to 0 on every non-GRANT-accumulating evaluation.
* Priority order: SPOOF > UNKNOWN > DENY > accumulate-toward-GRANT > GRANT.
  A high similarity score never overrides a failed anti-spoof check.
"""

from __future__ import annotations

from enum import Enum, auto

# ---------------------------------------------------------------------------
# Module-level thresholds — named constants so unit tests can import them
# directly and callers can override via DecisionEngine.__init__ kwargs.
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLD: float = 0.5
"""Minimum cosine-similarity score to consider an identity match.

Matches ``recognition.similarity_threshold`` in configs/config.yaml. See the
capstone parameter rationale in README.md (活體偵測與蜂鳴器設計決策)."""

REQUIRED_CONSECUTIVE_FRAMES: int = 4
"""Number of consecutive matching frames required before GRANT is issued.

Raised to 4 so the per-frame real-vs-photo gap is amplified across frames
(see README.md). Matches ``recognition.confirm_frames`` in config.yaml."""

LIVENESS_THRESHOLD: float = 0.3
"""Minimum MiniFASNet liveness score to pass the anti-spoof gate.

The AI pipeline converts this to a bool (``anti_spoof_pass``) before
calling :meth:`DecisionEngine.evaluate`, so this constant documents the
recommended threshold for the upstream pipeline configuration rather than
being evaluated inside DecisionEngine itself.
"""


# ---------------------------------------------------------------------------
# Decision enum
# ---------------------------------------------------------------------------


class Decision(Enum):
    """Five mutually exclusive access-control outcomes."""

    GRANT = auto()
    DENY = auto()
    UNKNOWN = auto()
    SPOOF = auto()
    IGNORE = auto()


# ---------------------------------------------------------------------------
# DecisionEngine
# ---------------------------------------------------------------------------


class DecisionEngine:
    """Access-control decision state machine.

    Parameters
    ----------
    similarity_threshold:
        Cosine-similarity cutoff for an identity match.
        Defaults to :data:`SIMILARITY_THRESHOLD` (0.5).
    required_frames:
        Number of consecutive qualifying frames before issuing
        :attr:`Decision.GRANT`.
        Defaults to :data:`REQUIRED_CONSECUTIVE_FRAMES` (4).

    Examples
    --------
    Typical orchestrator usage::

        engine = DecisionEngine()
        decision = engine.evaluate(
            similarity=0.91,
            anti_spoof_pass=True,
            face_in_db=True,
        )
    """

    def __init__(
        self,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        required_frames: int = REQUIRED_CONSECUTIVE_FRAMES,
    ) -> None:
        """Initialise thresholds and reset the internal frame counter."""
        if similarity_threshold <= 0.0 or similarity_threshold > 1.0:
            msg = f"similarity_threshold must be in (0, 1]; got {similarity_threshold}"
            raise ValueError(msg)
        if required_frames < 1:
            msg = f"required_frames must be >= 1; got {required_frames}"
            raise ValueError(msg)

        self._similarity_threshold: float = similarity_threshold
        self._required_frames: int = required_frames
        self._consecutive_frames: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def consecutive_frames(self) -> int:
        """Current consecutive-match frame count (read-only).

        Exposed as a property so tests can inspect internal state without
        breaking encapsulation.
        """
        return self._consecutive_frames

    def evaluate(
        self,
        similarity: float,
        anti_spoof_pass: bool,
        face_in_db: bool,
    ) -> Decision:
        """Evaluate one frame of AI pipeline output and return a Decision.

        Priority order (highest first)
        --------------------------------
        1. ``anti_spoof_pass`` is False → :attr:`Decision.SPOOF`
        2. ``face_in_db`` is False      → :attr:`Decision.UNKNOWN`
        3. ``similarity`` < threshold   → :attr:`Decision.DENY`
        4. All conditions met but frame count < required
                                        → accumulate; return :attr:`Decision.IGNORE`
        5. Frame count reaches required → :attr:`Decision.GRANT`

        Parameters
        ----------
        similarity:
            Cosine-similarity score ∈ [0, 1] from MobileFaceNet against
            the closest enrolled identity embedding.
        anti_spoof_pass:
            ``True`` when MiniFASNet liveness score exceeds its threshold.
        face_in_db:
            ``True`` when the closest enrolled identity has
            ``similarity >= similarity_threshold``.

        Returns
        -------
        Decision
            One of GRANT / DENY / UNKNOWN / SPOOF / IGNORE.
        """
        # ── Priority 1: anti-spoof failure overrides everything ──────────
        if not anti_spoof_pass:
            self._reset()
            return Decision.SPOOF

        # ── Priority 2: face not enrolled ────────────────────────────────
        if not face_in_db:
            self._reset()
            return Decision.UNKNOWN

        # ── Priority 3: similarity below threshold ────────────────────────
        if similarity < self._similarity_threshold:
            self._reset()
            return Decision.DENY

        # ── Priority 4/5: accumulate consecutive-frame counter ────────────
        self._consecutive_frames += 1

        if self._consecutive_frames >= self._required_frames:
            self._reset()
            return Decision.GRANT

        # Frame count not yet reached — keep accumulating, signal IGNORE
        # so the orchestrator takes no actuator action this frame.
        return Decision.IGNORE

    def ignore(self) -> Decision:
        """Signal that HC-SR04 timed out with no face detected.

        Resets the consecutive-frame counter and returns
        :attr:`Decision.IGNORE`.  Called by the orchestrator when the
        HC-SR04 gate fires but no face is detected within the 1-second
        window.

        Returns
        -------
        Decision
            Always :attr:`Decision.IGNORE`.
        """
        self._reset()
        return Decision.IGNORE

    def reset(self) -> None:
        """Explicitly reset the consecutive-frame counter to zero.

        Useful when the orchestrator detects a scene change (e.g. the
        person walked away mid-verification) and wants to restart the
        accumulation without calling :meth:`evaluate`.
        """
        self._reset()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        """Reset the consecutive-frame counter to zero."""
        self._consecutive_frames = 0
