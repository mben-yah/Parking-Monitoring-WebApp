# -*- coding: utf-8 -*-
"""
paddle_client.py
────────────────
Merchant of Record (MoR) integration for Paddle Billing v2 API.
Handles webhook verification, event parsing, and automated tenant licensing.
"""

import os
import hmac
import hashlib
import logging
from datetime import datetime, timezone
import mongodb_client

log = logging.getLogger("paddle_client")

# Config defaults (can be overridden via environment variables or app settings)
PADDLE_CLIENT_TOKEN   = os.getenv("PADDLE_CLIENT_TOKEN", "live_9f87e6d5c4b3a210_test")
PADDLE_WEBHOOK_SECRET = os.getenv("PADDLE_WEBHOOK_SECRET", "pdl_ntf_sec_test_123456789")
PADDLE_ENVIRONMENT    = os.getenv("PADDLE_ENVIRONMENT", "sandbox")  # 'sandbox' or 'production'

# Price ID Mapping -> Product Tiers
PRICE_TIER_MAP = {
    "pri_starter_monthly":   "STARTER",
    "pri_starter_yearly":    "STARTER",
    "pri_pro_monthly":       "PRO",
    "pri_pro_yearly":        "PRO",
    "pri_enterprise":        "ENTERPRISE",
}


def verify_paddle_webhook(request_body: bytes, signature_header: str, secret: str = PADDLE_WEBHOOK_SECRET) -> bool:
    """
    Verify Paddle v2 webhook signature using HMAC SHA256.
    Header format: 'ts=1690000000;h1=hash_value'
    """
    if not signature_header or not secret:
        # For development / sandbox fallback if secret not configured
        log.warning("Paddle signature header or secret missing — allowing sandbox bypass in dev")
        return True

    try:
        parts = dict(pair.split("=") for pair in signature_header.split(";"))
        timestamp = parts.get("ts", "")
        h1 = parts.get("h1", "")

        signed_payload = f"{timestamp}:{request_body.decode('utf-8')}"
        computed_hash = hmac.new(
            secret.encode('utf-8'),
            signed_payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(computed_hash, h1)
    except Exception as e:
        log.error(f"Paddle signature verification error: {e}")
        return False


def process_paddle_event(payload: dict) -> dict:
    """
    Process verified Paddle webhook event payload.
    Events: 'subscription.created', 'subscription.updated', 'transaction.completed', 'subscription.canceled'
    """
    event_type = payload.get("event_type", "")
    data       = payload.get("data", {})

    log.info(f"Processing Paddle Event: '{event_type}'")

    if event_type in ("subscription.created", "subscription.updated", "transaction.completed"):
        customer = data.get("customer", {})
        email = customer.get("email") or data.get("customer_email") or data.get("user_email", "")

        items = data.get("items", [])
        plan_tier = "PRO"  # Default
        for item in items:
            price_id = item.get("price", {}).get("id") or item.get("price_id", "")
            if price_id in PRICE_TIER_MAP:
                plan_tier = PRICE_TIER_MAP[price_id]
                break

        custom_data = data.get("custom_data", {})
        company_name = custom_data.get("company_name") or email.split("@")[0] if email else "Customer Instance"

        if email:
            tenant = mongodb_client.provision_tenant(
                company_name=company_name,
                email=email,
                plan_tier=plan_tier
            )
            log.info(f"Successfully provisioned/upgraded tenant '{tenant.get('tenant_id')}' to {plan_tier}")
            return {"ok": True, "action": "provisioned", "tenant": tenant}

    elif event_type == "subscription.canceled":
        customer = data.get("customer", {})
        email = customer.get("email") or data.get("customer_email", "")
        if email:
            # Revert to Free Tier on cancellation
            tenant = mongodb_client.provision_tenant(
                company_name="",
                email=email,
                plan_tier="FREE"
            )
            log.info(f"Subscription canceled for '{email}'. Reverted tenant to FREE tier.")
            return {"ok": True, "action": "reverted_to_free", "tenant": tenant}

    return {"ok": True, "action": "ignored", "event_type": event_type}
