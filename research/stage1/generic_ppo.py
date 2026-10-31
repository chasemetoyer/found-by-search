"""Compatibility entrypoint for the corrected archived PPO adapter.

Generic rewards use intrinsic.kind='generic_homeostatic'. All other arms use
the same corrected plumbing through stage1.corrected_ppo; there is no path
that silently restores the audited detector defects.
"""
from stage1.corrected_ppo import load_ppo, main


if __name__ == "__main__":
    main()
