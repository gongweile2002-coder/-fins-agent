# ruff: noqa
# The crypto data already provides dollar trading volume as usd_volume.
# Keep a common feature name so the rest of the script can use dollar_volume.
crypto_panel["dollar_volume"] = crypto_panel["usd_volume"]
