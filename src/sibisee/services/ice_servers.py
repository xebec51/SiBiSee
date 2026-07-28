from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)
DEFAULT_STUN_SERVERS = [{"urls": ["stun:stun.l.google.com:19302"]}]


def get_ice_servers(secrets: Any, twilio_client_factory: Any | None = None) -> list[dict[str, Any]]:
    try:
        twilio_secrets = secrets.get("twilio", {}) if hasattr(secrets, "get") else secrets["twilio"]
        account_sid = twilio_secrets["ACCOUNT_SID"]
        auth_token = twilio_secrets["AUTH_TOKEN"]
        if twilio_client_factory is None:
            from twilio.rest import Client

            twilio_client_factory = Client
        client = twilio_client_factory(account_sid, auth_token)
        token = client.tokens.create()
        return list(token.ice_servers)
    except Exception:
        LOGGER.info("Twilio ICE server lookup failed; falling back to public STUN.", exc_info=True)
        return DEFAULT_STUN_SERVERS
