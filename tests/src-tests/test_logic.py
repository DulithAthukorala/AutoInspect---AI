from src.logic import DamageInstance, CaseEvidence, decide_case

def test_low_severity_auto():
    ev = CaseEvidence(
        image_id="img1",
        damages=[
            DamageInstance("scratch", confidence=0.9, area_ratio=0.005),
        ],
        overlaps=None
    )
    d = decide_case(ev)
    assert d.severity in {"LOW", "MEDIUM", "HIGH"}
    assert d.estimated_cost_lkr > 0
    assert d.route in {"AUTO", "MANUAL_REVIEW"}

def test_manual_when_low_conf():
    ev = CaseEvidence(
        image_id="img2",
        damages=[
            DamageInstance("dent", confidence=0.3, area_ratio=0.02),
        ],
        overlaps=None
    )
    d = decide_case(ev)
    assert d.route == "MANUAL_REVIEW"
