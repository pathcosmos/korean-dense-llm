"""Public checkout support. Optional original-run integrations are explicitly reported."""
import os
from pathlib import Path
import tempfile
import pytest
ROOT=Path(__file__).resolve().parents[1]
PRODUCTION_TESTS={
 'test_original_validated_copy_settings',
 'test_original_3b_optimizer_metadata',
 'test_actual_main_marker_without_target_cannot_pass_final_gate',
 'test_single_repository_release_download_paths_and_retry',
 'test_release_assembly_and_retry_keep_13_languages_and_pins',
 'test_final_native_package_retains_exact_732_inputs_and_reuses_release_on_retry',
 'test_current_prepared_plan_bindings_and_worker_tmpdir_admission',
 'test_actual_parent_paths_protected_and_current_bindings_preparable',
 'test_sigterm_stops_owned_worker_only_and_preserves_incomplete_evidence',
}
TWO_FILESYSTEM_MODULES={'test_tiered_checkpoint_retention.py','test_checkpoint_retention_audit.py','test_retention3_policy.py'}

def pytest_addoption(parser):
    parser.addoption('--run-original-integrations',action='store_true',default=False,help='Require locally restored original run manifests; not included in the source-only checkout')

def pytest_sessionstart(session):
    runs=ROOT/'runs'
    if runs.is_symlink():raise RuntimeError('Tests require a checkout-owned runs directory, not a symlink to production')
    runs.mkdir(exist_ok=True)

def pytest_collection_modifyitems(config,items):
    same_device=ROOT.stat().st_dev==Path(tempfile.gettempdir()).stat().st_dev
    for item in items:
        if item.name in PRODUCTION_TESTS and not config.getoption('--run-original-integrations'):
            item.add_marker(pytest.mark.skip(reason='Original-run integration: requires excluded production dataset release/active training manifests; use --run-original-integrations only in a restored test copy'))
        if Path(str(item.path)).name in TWO_FILESYSTEM_MODULES and same_device:
            item.add_marker(pytest.mark.skip(reason='Retention safety requires two distinct filesystems; put TMPDIR on a separate device to exercise this CPU fixture'))
