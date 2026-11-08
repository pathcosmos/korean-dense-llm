"""Capacity planning for bounded, disposable local checkpoint history."""
import math


def history_growth(policy, used):
    remaining=max(0,policy['local_budget_bytes']-used)
    if policy.get('adaptive_local_pruning'):
        return min(remaining,math.ceil(policy['expected_checkpoint_bytes']*1.1)+2**30)
    return remaining


def enforce_capacity(manager, *, busy=False):
    try:
        manager.budgets()
    except RuntimeError:
        if not manager.policy.get('adaptive_local_pruning') or busy:
            raise  # Never compete with a copy/deletion transaction under pressure.
        with manager.locked():
            manager.make_local_room()
            manager.budgets()
        manager.maintain()  # Publish the resulting count and journal state.
