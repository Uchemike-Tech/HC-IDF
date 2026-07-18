import numpy as np
import pandas as pd
from collections import defaultdict


class MITMDetectionModule:
    def __init__(self, config: dict):
        self.config = config["mitm"]
        self.arp_table = {}
        self.dns_query_log = defaultdict(list)
        self.session_baselines = {}

    def check_arp_spoofing(self, mac: str, ip: str, timestamp: float) -> bool:
        alert = False
        if ip in self.arp_table:
            if self.arp_table[ip] != mac:
                alert = True
        else:
            self.arp_table[ip] = mac
        return alert

    def check_dns_poisoning(self, query_id: int, response_id: int,
                            ttl: float, baseline_ttl: float) -> bool:
        deviation = abs(ttl - baseline_ttl)
        threshold = self.config["dns"]["max_response_deviation"]
        if deviation > threshold or query_id != response_id:
            return True
        return False

    def check_session_hijacking(self, src_ip: str, rtt: float,
                                tcp_seq_jump: int) -> bool:
        if src_ip not in self.session_baselines:
            self.session_baselines[src_ip] = {"rtt_mean": rtt, "rtt_std": 0, "count": 1}
            return False

        baseline = self.session_baselines[src_ip]
        rtt_std_threshold = self.config["session"]["rtt_std_threshold"]
        rtt_deviation = abs(rtt - baseline["rtt_mean"])

        if baseline["count"] > 10 and rtt_deviation > rtt_std_threshold * max(baseline["rtt_std"], 1e-6):
            return True

        n = baseline["count"]
        baseline["rtt_mean"] = (baseline["rtt_mean"] * n + rtt) / (n + 1)
        baseline["rtt_std"] = np.sqrt(
            (baseline["rtt_std"] ** 2 * n + (rtt - baseline["rtt_mean"]) ** 2) / (n + 1)
        )
        baseline["count"] += 1
        return False

    def analyze_packet(self, packet: dict) -> dict:
        alerts = []
        if packet.get("type") == "arp":
            flagged = self.check_arp_spoofing(packet["mac"], packet["ip"], packet["timestamp"])
            if flagged:
                alerts.append("ARP_SPOOFING")
        elif packet.get("type") == "dns":
            flagged = self.check_dns_poisoning(
                packet["query_id"], packet["response_id"],
                packet["ttl"], packet.get("baseline_ttl", 300),
            )
            if flagged:
                alerts.append("DNS_POISONING")
        elif packet.get("type") == "tcp":
            flagged = self.check_session_hijacking(
                packet["src_ip"], packet["rtt"], packet.get("tcp_seq_jump", 0),
            )
            if flagged:
                alerts.append("SESSION_HIJACKING")

        return {
            "mitm_alert": len(alerts) > 0,
            "mitm_type": alerts if alerts else None,
            "mitm_score": len(alerts) / 3.0,
        }
