"""Demonstration of HybridMemoryAgent with 5 representative queries.

Demonstrates:
  1. Simple lookup (vector hit): "Tôi đã đọc gì về Kubernetes?"
  2. Profile-needed: "Recommend đọc gì tiếp theo"
  3. Fresh activity: "Tôi đang quan tâm gì gần đây?"
  4. Paraphrase (vector hit): "Tài liệu về tự động mở rộng hạ tầng?"
  5. Mixed (episodic + profile): "Cho tôi tóm tắt bảo mật đám mây"
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bonus.agent import HybridMemoryAgent


def main() -> int:
    print("================================================================")
    print("       HYBRID AI MEMORY SYSTEM DEMO — VIETNAMESE ASSISTANT      ")
    print("================================================================")

    agent = HybridMemoryAgent(feast_repo_path=ROOT / "app" / "feast_repo")

    # Seed some sample episodic memories for user u_001
    user_id = "u_001"
    print(f"\n[Step 1] Ingesting sample episodic memories for user: {user_id}...")
    sample_notes = [
        "Kubernetes (K8s) là nền tảng điều phối container mã nguồn mở giúp tự động hóa việc triển khai và quản lý.",
        "Kiến trúc microservices yêu cầu triển khai autoscaling dựa trên CPU/Memory metrics và HPA.",
        "Bảo mật đám mây (Cloud Security) cần kết hợp Zero Trust Architecture, mã hóa KMS và RBAC chặt chẽ.",
        "Hệ thống CI/CD pipeline tự động hóa kiểm thử và triển khai với ArgoCD trên hạ tầng Kubernetes.",
    ]

    for note in sample_notes:
        agent.remember(note, user_id=user_id)
    print(f"  Ingested {len(sample_notes)} notes into episodic memory.")

    # 5 Demo queries
    queries = [
        ("Query 1 [Vector Lookup]", "Tôi đã đọc gì về Kubernetes?"),
        ("Query 2 [Profile Context]", "Recommend đọc gì tiếp theo dựa trên sở thích của tôi"),
        ("Query 3 [Recent Activity]", "Tôi đang quan tâm gì và có hoạt động gì gần đây?"),
        ("Query 4 [Paraphrase Search]", "Tài liệu về tự động mở rộng hạ tầng theo tải người dùng"),
        ("Query 5 [Mixed Hybrid]", "Cho tôi tóm tắt bảo mật đám mây và khuyến nghị theo ngôn ngữ ưu tiên"),
    ]

    print("\n[Step 2] Executing 5 Representative Queries:\n")
    for title, q in queries:
        print(f"--- {title} ---")
        print(f"User Query: {q!r}")
        context = agent.recall(q, user_id=user_id, top_k=2)
        print(context)
        print()

    print("Demo completed successfully. Exit code 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
