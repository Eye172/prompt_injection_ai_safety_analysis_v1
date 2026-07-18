"""API-key auth, rate limiting and credit accounting (spec 2.4 step 4)."""

from __future__ import annotations

import logging
import unittest

from conftest import BENIGN_EN, api_client

logging.disable(logging.INFO)

PAYLOAD = {"text": BENIGN_EN}


class AuthTests(unittest.TestCase):
    def test_open_when_no_keys_configured(self):
        with api_client() as client:
            self.assertEqual(client.post("/v1/detect", json=PAYLOAD).status_code, 200)

    def test_key_required_once_configured(self):
        with api_client(api_keys="secret-a,secret-b") as client:
            self.assertEqual(client.post("/v1/detect", json=PAYLOAD).status_code, 401)

            bad = client.post(
                "/v1/detect", json=PAYLOAD, headers={"X-API-Key": "wrong"}
            )
            self.assertEqual(bad.status_code, 401)
            self.assertEqual(bad.json()["error"]["code"], "invalid_api_key")

            for key in ("secret-a", "secret-b"):
                ok = client.post(
                    "/v1/detect", json=PAYLOAD, headers={"X-API-Key": key}
                )
                self.assertEqual(ok.status_code, 200, key)

    def test_health_stays_reachable_without_a_key(self):
        """Probes must not need credentials, or orchestrators cannot check us."""
        with api_client(api_keys="secret-a") as client:
            self.assertEqual(client.get("/v1/health").status_code, 200)


class RateLimitTests(unittest.TestCase):
    def test_limit_exhausts_and_reports_credits(self):
        with api_client(rate_limit_requests=3, rate_limit_window_seconds=60) as client:
            remaining = []
            for _ in range(3):
                resp = client.post("/v1/detect", json=PAYLOAD)
                self.assertEqual(resp.status_code, 200)
                remaining.append(resp.json()["meta"]["credits_remaining"])

            self.assertEqual(remaining, [2, 1, 0])

            blocked = client.post("/v1/detect", json=PAYLOAD)
            self.assertEqual(blocked.status_code, 429)
            self.assertEqual(blocked.json()["error"]["code"], "rate_limit_exceeded")
            self.assertIn("Retry-After", blocked.headers)

    def test_credits_consumed_reported(self):
        with api_client(rate_limit_requests=10) as client:
            meta = client.post("/v1/detect", json=PAYLOAD).json()["meta"]
        self.assertEqual(meta["credits_consumed"], 1)

    def test_limits_are_per_key_not_global(self):
        with api_client(
            api_keys="key-one,key-two", rate_limit_requests=2
        ) as client:
            for _ in range(2):
                client.post("/v1/detect", json=PAYLOAD, headers={"X-API-Key": "key-one"})
            exhausted = client.post(
                "/v1/detect", json=PAYLOAD, headers={"X-API-Key": "key-one"}
            )
            self.assertEqual(exhausted.status_code, 429)

            # A different caller must be unaffected by the first one's usage.
            other = client.post(
                "/v1/detect", json=PAYLOAD, headers={"X-API-Key": "key-two"}
            )
            self.assertEqual(other.status_code, 200)


class PrivacyTests(unittest.TestCase):
    def test_request_text_is_not_logged_by_default(self):
        """Spec 2.4 step 6: store the decision, never the client's text."""
        import json

        from api.config import Settings
        from api.observability import log_detection
        from api.schemas import DetectionData, Label

        records = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        logger = logging.getLogger("api.detections")
        handler = Capture()
        logger.addHandler(handler)
        logging.disable(logging.NOTSET)
        try:
            secret = "my password is hunter2 and my card is 4111111111111111"
            result = DetectionData(
                is_injection=False, score=0.1, label=Label.benign, language_used="en"
            )
            log_detection(
                "req_test", [result], 5, Settings(backend="stub"),
                endpoint="/v1/detect", texts=[secret],
            )
        finally:
            logger.removeHandler(handler)
            logging.disable(logging.INFO)

        self.assertEqual(len(records), 1)
        self.assertNotIn("hunter2", records[0])
        self.assertNotIn("4111111111111111", records[0])
        # The decision itself is still recorded, for drift monitoring.
        payload = json.loads(records[0])
        self.assertEqual(payload["detections"][0]["label"], "benign")


if __name__ == "__main__":
    unittest.main(verbosity=2)
