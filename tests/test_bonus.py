"""Unit tests for the Bonus Challenge: HybridMemoryAgent.

Verifies:
  1. remember() chunks and indexes text.
  2. recall() retrieves relevant memory and returns structured context.
  3. Strict Multi-user Isolation (User A's private memory cannot be recalled by User B).
"""
from __future__ import annotations

import pytest
from bonus.agent import HybridMemoryAgent


def test_agent_remembers_and_recalls():
    agent = HybridMemoryAgent()
    agent.remember(
        "Kubernetes cluster autoscaler automatically increases the number of nodes.",
        user_id="user_alice",
    )

    ctx = agent.recall("How does autoscaler work?", user_id="user_alice", top_k=1)
    assert "Kubernetes cluster autoscaler" in ctx
    assert "user_alice" in ctx


def test_multi_user_isolation_blocks_cross_user_leakage():
    """Security Gate: User Bob must NEVER see Alice's private notes."""
    agent = HybridMemoryAgent()

    agent.remember(
        "TOP_SECRET_PASSWORD_123: Alice confidential project roadmap",
        user_id="user_alice",
    )
    agent.remember(
        "Public knowledge: Python 3.12 release features",
        user_id="user_bob",
    )

    # Bob asks about secret password
    bob_ctx = agent.recall("What is the TOP_SECRET_PASSWORD?", user_id="user_bob")
    assert "TOP_SECRET_PASSWORD_123" not in bob_ctx
    assert "Alice confidential" not in bob_ctx

    # Alice asks about her password -> should find it
    alice_ctx = agent.recall("What is my secret password?", user_id="user_alice")
    assert "TOP_SECRET_PASSWORD_123" in alice_ctx
