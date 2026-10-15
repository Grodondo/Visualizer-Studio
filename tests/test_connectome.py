"""Checks for the connectome video: the export agrees with itself and with the renderer, and every sentence the video
says about the data is true for the exported numbers (if one fails after a data change, change the sentence, not the test)."""
import json

import numpy as np
import pytest

from compute import connectome, fly_render
from connectome import GF_COLOR, TYPE_COLOR
from studio.fly import FRAMES_DIR, FlyTrace, camera


@pytest.fixture(scope="module")
def trace() -> FlyTrace:
    return FlyTrace.load(connectome.TRACE_PATH)


def test_dataset(trace):
    d = trace["dataset"]
    assert d["license"] == "CC BY 4.0" and "v2.1.0" in d["annotations"]
    assert d["connections_generation"] == connectome.CONNECTIONS["gcs_generation"]
    assert trace["totals"]["neurons"] == 139255  # Dorkenwald et al. 2024


def test_brain_size_matches_the_em_volume(trace):
    """Units check: Zheng et al. 2018 imaged sections ~750 um wide in a 995 x 537 x 283 um volume."""
    e = trace["extent_um"]
    width, height, depth = (e[k][1] - e[k][0] for k in ("x", "y", "z"))
    assert 700 < width < 995 and 300 < height < 537 and 250 < depth < 283
    assert round(depth / 0.04, -3) == 7000  # "cut into about 7,000 slices"


def test_points_file(trace):
    p = np.load(connectome.POINTS_PATH)
    assert p["xyz"].shape == (trace["totals"]["neurons"], 3)
    counts = np.bincount(p["cls"], minlength=len(p["class_names"]))
    assert dict(zip(p["class_names"], counts.tolist())) == trace["totals"]["classes"]


def test_claim_optic_lobes_hold_more_than_half(trace):
    t = trace["totals"]
    assert t["classes"]["optic"] / t["neurons"] > 0.5


def test_first_group(trace):
    g = trace["first_group"]
    ids = {x["id"] for x in g["neurons"]}
    assert len(ids) == connectome.FIRST_GROUP and g["neurons"][0]["type"] == "DNp01"
    assert all(e["pre"] in ids and e["post"] in ids and e["syn"] >= 5 for e in g["edges"])


def test_claim_looming_detectors_are_the_largest_inputs(trace):
    f = trace["focus"]
    assert [t["type"] for t in f["input_types"][:2]] == ["LPLC2", "LC4"]
    assert f["type"] == "DNp01" and f["side"] == "right" and f["other_side"]["side"] == "left"


def test_synapse_file_agrees_with_connection_table(trace):
    f, ext = trace["focus"], trace["synapse_extract"]
    assert ext["focus_inputs"] == f["input_synapses"]
    by_type = {t["type"]: t["synapses"] for t in f["input_types"]}
    assert ext["focus_inputs_by_type"] == {t: by_type[t] for t in connectome.LOOMING_TYPES}
    z = np.load(connectome.SYNAPSES_PATH)
    assert int((z["post"] == int(f["id"])).sum()) == ext["focus_inputs"]


def test_claim_hub_contacts_many_neurons(trace):
    """'only N% of these synapses land on the giant fiber itself'"""
    ext = trace["synapse_extract"]
    for t in connectome.LOOMING_TYPES:
        assert ext["focus_inputs_by_type"][t] / ext["looming_outputs"][t] < 0.25


def test_claim_transmitter_predictions_disagree_with_low_confidence(trace):
    a, b = trace["focus"], trace["focus"]["other_side"]
    assert a["predicted_nt"] != b["predicted_nt"]
    assert a["predicted_nt_confidence"] < 0.5 and b["predicted_nt_confidence"] < 0.5


@pytest.mark.parametrize("angle", [0, 37, 90, 180, 300])
def test_scene_camera_matches_renderer(angle):
    assert np.allclose(camera(angle), fly_render.camera(angle))


RENDERED = (FRAMES_DIR / "manifest.json").exists()


@pytest.mark.skipif(not RENDERED, reason="fly frames not rendered (uv run python -m compute.fly_render)")
def test_rendered_frames(trace):
    m = json.loads((FRAMES_DIR / "manifest.json").read_text(encoding="utf-8"))
    for angle in m["turntable_angles"]:
        assert (FRAMES_DIR / m["turntable"].format(angle=angle)).exists()
    for angle in m["circuit_turntable"]["angles"]:
        assert (FRAMES_DIR / m["circuit_turntable"]["files"].format(angle=angle)).exists()
    # the colors the scene uses for its labels are the colors of the synapse images
    colors = m["zoom"]["synapse_colors"]
    assert GF_COLOR.to_hex().lower() == colors["GF"].lower()
    for t, c in TYPE_COLOR.items():
        assert c.to_hex().lower() == colors[t].lower()
    assert m["zoom"]["counts"]["GF"] == trace["focus"]["input_synapses"]
