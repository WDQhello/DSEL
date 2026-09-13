import torch

from pace_clrf.utils.checkpoint import normalize_state_dict


def test_legacy_checkpoint_names_are_remapped():
    state_dict = {
        "module.fpsm.freq_mask": torch.ones(2),
        "module.mil_head.fc1.weight": torch.ones(2, 2),
    }

    normalized = normalize_state_dict(state_dict, remap_legacy_names=True)

    assert "pace.freq_mask" in normalized
    assert "clrf.fc1.weight" in normalized
