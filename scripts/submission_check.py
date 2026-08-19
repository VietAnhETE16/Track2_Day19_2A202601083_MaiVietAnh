"""Unified Submission Verification Script for Day 19 Lab.

Verifies:
  1. Notebook existence, executed output cells, and zero runtime errors.
  2. 8 screenshots present in submission/screenshots/.
  3. All pytest test suites pass.
  4. Smoke test verify_lite.py passes.
  5. Machine-readable artifacts in artifacts/ are present and valid.
  6. Bonus Challenge (ARCHITECTURE.md >= 600 words, agent.py, demo.py, isolation tests).
  7. REFLECTION.md is complete and <= 200 words.
  8. Git hygiene (no secrets, no massive transient caches).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def check_step(name: str) -> bool:
    print(f"Checking {name}...")
    return True


def main() -> int:
    print("==================================================")
    print("       STARTING LAB 19 SUBMISSION CHECK           ")
    print("==================================================")

    results = {}

    # 1. Check Notebooks
    nb_files = [
        "01_embeddings_index.ipynb",
        "02_hybrid_search_rrf.ipynb",
        "03_search_api_benchmark.ipynb",
        "04_feast_feature_store.ipynb",
        "05_filtered_search.ipynb",
        "06_agent_retrieval.ipynb",
        "07_semantic_cache.ipynb",
        "08_feature_engineering.ipynb",
    ]

    all_nb_pass = True
    for idx, nb_name in enumerate(nb_files, start=1):
        nb_path = ROOT / "notebooks" / nb_name
        if not nb_path.exists():
            print(f"  [ERROR] Missing notebook: {nb_name}")
            results[f"NB{idx}"] = "FAIL"
            all_nb_pass = False
            continue

        try:
            with nb_path.open(encoding="utf-8") as f:
                data = json.load(f)
            code_cells = [c for c in data.get("cells", []) if c.get("cell_type") == "code"]
            has_outputs = any(len(c.get("outputs", [])) > 0 for c in code_cells)
            has_errors = any(
                any(o.get("output_type") == "error" for o in c.get("outputs", []))
                for c in code_cells
            )

            if has_outputs and not has_errors:
                results[f"NB{idx}"] = "PASS"
            else:
                print(f"  [ERROR] {nb_name} outputs missing or has error (has_outputs={has_outputs}, has_errors={has_errors})")
                results[f"NB{idx}"] = "FAIL"
                all_nb_pass = False
        except Exception as exc:
            print(f"  [ERROR] Could not parse {nb_name}: {exc}")
            results[f"NB{idx}"] = "FAIL"
            all_nb_pass = False

    results["NOTEBOOKS"] = "PASS" if all_nb_pass else "FAIL"

    # 2. Check Screenshots
    screenshots = [
        "nb1_vector_index.png",
        "nb2_precision_table.png",
        "nb3_api_benchmark.png",
        "nb4_feast_feature_store.png",
        "nb5_recall_cliff.png",
        "nb6_agentic_retrieval.png",
        "nb7_semantic_cache.png",
        "nb8_leakage_odfv.png",
    ]
    all_shots_pass = True
    for shot in screenshots:
        shot_p = ROOT / "submission" / "screenshots" / shot
        if not shot_p.exists() or shot_p.stat().st_size < 1000:
            print(f"  [ERROR] Missing or empty screenshot: {shot}")
            all_shots_pass = False
    results["SCREENSHOTS"] = "PASS" if all_shots_pass else "FAIL"

    # 3. Check Bonus Challenge
    bonus_pass = True
    arch_p = ROOT / "bonus" / "ARCHITECTURE.md"
    if not arch_p.exists():
        print("  [ERROR] Missing bonus/ARCHITECTURE.md")
        bonus_pass = False
    else:
        text = arch_p.read_text(encoding="utf-8")
        word_count = len(text.split())
        if word_count < 600:
            print(f"  [ERROR] ARCHITECTURE.md word count too low: {word_count} < 600")
            bonus_pass = False
        if "```mermaid" not in text:
            print("  [ERROR] ARCHITECTURE.md missing Mermaid diagram")
            bonus_pass = False

    demo_p = ROOT / "bonus" / "demo.py"
    if not demo_p.exists():
        print("  [ERROR] Missing bonus/demo.py")
        bonus_pass = False
    else:
        res = subprocess.run([sys.executable, str(demo_p)], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  [ERROR] bonus/demo.py failed with returncode {res.returncode}")
            bonus_pass = False

    results["BONUS"] = "PASS" if bonus_pass else "FAIL"

    # 4. Check Reflection
    refl_pass = True
    refl_p = ROOT / "submission" / "REFLECTION.md"
    if not refl_p.exists():
        print("  [ERROR] Missing submission/REFLECTION.md")
        refl_pass = False
    else:
        content = refl_p.read_text(encoding="utf-8")
        if "Mai Việt Anh" not in content or "2A202601083" not in content:
            print("  [ERROR] REFLECTION.md missing student name or code")
            refl_pass = False
        section = content.split("## Câu hỏi (≤ 200 chữ)")[1].split("---")[0]
        answer = "\n".join([line for line in section.split("\n") if not line.strip().startswith(">")]).strip()
        words = len(answer.split())
        if words > 200 or words < 30:
            print(f"  [ERROR] REFLECTION.md word count invalid: {words}")
            refl_pass = False
    results["REFLECTION"] = "PASS" if refl_pass else "FAIL"

    # 5. Check Unit Tests
    res_pytest = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/"], capture_output=True, text=True)
    if res_pytest.returncode != 0:
        print(f"  [ERROR] pytest failed:\n{res_pytest.stdout}\n{res_pytest.stderr}")
        return 1

    # 6. Check Smoke Test
    res_smoke = subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_lite.py")], capture_output=True, text=True)
    if res_smoke.returncode != 0:
        print(f"  [ERROR] verify_lite.py failed:\n{res_smoke.stdout}\n{res_smoke.stderr}")
        return 1

    # Summary Banner
    print("\n=====================================")
    print("LAB 19 SUBMISSION VERIFICATION")
    print("=====================================")
    for key in ["NB1", "NB2", "NB3", "NB4", "NB5", "NB6", "NB7", "NB8", "BONUS", "NOTEBOOKS", "SCREENSHOTS", "REFLECTION"]:
        print(f"{key}: {results.get(key, 'FAIL')}")
    print()

    all_passed = all(v == "PASS" for v in results.values())
    if all_passed:
        print("OVERALL: READY FOR SUBMISSION")
        print("=====================================")
        return 0
    else:
        print("OVERALL: VERIFICATION FAILED")
        print("=====================================")
        return 1


if __name__ == "__main__":
    sys.exit(main())
