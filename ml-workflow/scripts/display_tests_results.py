
import json
import sys

def display_results(json_file):
    try:
        with open(json_file) as f:
            data = json.load(f)
        
        print("========================= RÉSUMÉ DES TESTS =========================")
        passed = data.get("summary", {}).get("passed", 0)
        total = data.get("summary", {}).get("total", 0)
        duration = data.get("duration", 0)

        print(f"✅ Tests réussis: {passed}/{total}")
        print(f"⏱️ Durée totale: {duration:.2f} secondes\n")

        print("========================= DÉTAILS DES TESTS =========================")
        for test in data.get("tests", []):
            nodeid = test.get("nodeid", "")
            outcome = test.get("outcome", "")
            test_name = nodeid.split("::")[-1] if "::" in nodeid else nodeid
            status = "✓" if outcome == "passed" else "✗"
            print(f"{status} {test_name}")

        print("\n========================= AVERTISSEMENTS =========================")
        warnings = data.get("warnings", [])
        if warnings:
            print(f"⚠️ {len(warnings)} avertissement(s) détecté(s)")
        else:
            print("Aucun avertissement")
    except Exception as e:
        print(f"Erreur lors de l'analyse du rapport: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        display_results(sys.argv[1])
    else:
        print("Usage: python display_tests_results.py /chemin/vers/resultats.json")
        sys.exit(1)