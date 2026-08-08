"""Research modules — standalone study CLIs, NOT wired into the live path.

Kept importable so their studies stay reproducible, but nothing in the
production pipeline (monitor / briefing / backtest) imports from here.

    portfolio         V1.4 multi-sleeve study (reversion + trend risk parity)
    gc_macro_model    candidate 60d-horizon GC macro ridge (pending fwd validation)
    gld_flow_features GLD dealer-flow feature extractor (awaiting chain history)
"""
