"""
On-chain data fetcher.

Production implementation requires Glassnode or CryptoQuant API keys.
Without a configured key this module returns neutral stub values so the
rest of the pipeline can run during development and paper trading setup.

Set GLASSNODE_API_KEY in .env to enable live on-chain metrics.

TODO (Phase 4): integrate Glassnode `/v1/metrics/transactions/transfers_volume_to_exchanges_sum`
                for exchange_net_flow_btc and SOPR.
"""

import os


async def fetch_onchain(glassnode_key: str | None = None) -> dict:
    """Fetch on-chain metrics.

    Returns:
        Dict with keys: exchange_net_flow_btc, whale_transactions_24h, sopr.
        Returns neutral stub values (0 / 0 / 1.0) when no API key is set.
    """
    resolved_key = glassnode_key or os.getenv("GLASSNODE_API_KEY")
    if not resolved_key:
        return {
            "exchange_net_flow_btc": 0.0,
            "whale_transactions_24h": 0,
            "sopr": 1.0,
        }

    # Glassnode integration placeholder — expand in Phase 4.
    raise NotImplementedError(
        "Glassnode integration is not yet implemented. "
        "Unset GLASSNODE_API_KEY to use stub values during development."
    )
