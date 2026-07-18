import pytest
import numpy as np
from src.detection.mitm_module import MITMDetectionModule


@pytest.fixture
def mitm_module():
    config = {"mitm": {"arp": {"cache_timeout": 300, "alert_threshold": 3},
                       "dns": {"max_response_deviation": 50},
                       "session": {"rtt_std_threshold": 2.5}}}
    return MITMDetectionModule(config)


class TestMITMModule:
    def test_arp_spoofing_detection(self, mitm_module):
        mitm_module.arp_table["192.168.1.1"] = "AA:BB:CC:DD:EE:01"
        result = mitm_module.check_arp_spoofing("AA:BB:CC:DD:EE:02", "192.168.1.1", 0.0)
        assert result is True

    def test_arp_no_spoofing(self, mitm_module):
        result = mitm_module.check_arp_spoofing("AA:BB:CC:DD:EE:01", "192.168.1.1", 0.0)
        assert result is False

    def test_dns_poisoning_detection(self, mitm_module):
        result = mitm_module.check_dns_poisoning(100, 101, 500, 300)
        assert result is True

    def test_session_hijacking_returns_false_initially(self, mitm_module):
        result = mitm_module.check_session_hijacking("10.0.0.1", 10.0, 0)
        assert result is False
