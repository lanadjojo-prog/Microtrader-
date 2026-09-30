import unittest

from config import Settings
from ctrader_client import CTraderClient, CTraderError


class CTraderSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        client = getattr(self, "client", None)
        if client is not None:
            await client.close()

    def make_client(self, **overrides):
        values = {
            "ctrader_environment": "demo",
            "ctrader_demo_only": True,
            "ctrader_oauth_scope": "accounts",
            "ctrader_client_id": "test-client",
            "ctrader_client_secret": "test-secret",
            "ctrader_redirect_uri": "https://example.test/callback",
        }
        values.update(overrides)
        self.client = CTraderClient(Settings(**values))
        return self.client

    async def test_demo_endpoint_uses_json_port(self):
        client = self.make_client()
        self.assertEqual(client.endpoint, "wss://demo.ctraderapi.com:5036")

    async def test_read_only_scope_is_default(self):
        client = self.make_client()
        url = client.authorization_url()
        self.assertIn("scope=accounts", url)
        self.assertNotIn("scope=trading", url)

    async def test_trading_scope_is_blocked_in_demo_only_mode(self):
        client = self.make_client()
        with self.assertRaises(CTraderError):
            client.authorization_url("trading")

    async def test_live_endpoint_is_blocked_in_demo_only_mode(self):
        client = self.make_client(ctrader_environment="live")
        with self.assertRaises(CTraderError):
            _ = client.endpoint

    async def test_invalid_environment_is_rejected(self):
        client = self.make_client(ctrader_environment="staging")
        with self.assertRaises(CTraderError):
            _ = client.endpoint

    def test_legacy_stock_execution_is_hard_disabled(self):
        self.assertFalse(
            Settings(paper=True, live_trading_enabled=True).can_trade
        )


if __name__ == "__main__":
    unittest.main()
