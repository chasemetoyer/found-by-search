"""Compatibility support for immutable pre-lock fixtures after registered mode switch."""
import pytest

@pytest.fixture(autouse=True)
def follow_rehearsal_mode_patch(request, monkeypatch):
    if request.module.__name__ == 'test_stage2_search_mode':
        import stage2.registration
        # The frozen fixture patches the defining module, but its test imported an alias.
        monkeypatch.setattr(request.module, 'mode_record', lambda: stage2.registration.mode_record())
