import unittest

from pku_qa.workflows.operations.check_web_stack import public_front_is_healthy


class PublicFrontHealthTests(unittest.TestCase):
    def test_basic_auth_challenge_is_unhealthy(self) -> None:
        self.assertFalse(public_front_is_healthy(401))

    def test_authenticated_response_is_healthy(self) -> None:
        self.assertTrue(public_front_is_healthy(200))

    def test_gateway_and_network_failures_are_unhealthy(self) -> None:
        for status in (0, 401, 404, 502, 503):
            self.assertFalse(public_front_is_healthy(status))
