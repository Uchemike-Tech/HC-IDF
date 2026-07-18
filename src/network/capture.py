import threading
import time
import numpy as np
from collections import defaultdict
from datetime import datetime

try:
    from scapy.all import sniff, IP, TCP, UDP, ARP, DNS, Ether
    SCAPY_AVAIL = True
except ImportError:
    SCAPY_AVAIL = False


class LiveCapture:
    def __init__(self, interface=None):
        self.interface = interface
        self.running = False
        self.thread = None
        self.stats = {
            "total_packets": 0,
            "tcp": 0, "udp": 0, "arp": 0, "dns": 0, "other": 0,
            "src_ips": defaultdict(int),
            "dst_ips": defaultdict(int),
            "ports": defaultdict(int),
            "start_time": None,
            "packet_history": [],
        }
        self.callback = None

    def _handle_packet(self, packet):
        if not self.running:
            return
        self.stats["total_packets"] += 1
        now = datetime.now()

        if packet.haslayer(TCP):
            self.stats["tcp"] += 1
            proto = "TCP"
        elif packet.haslayer(UDP):
            self.stats["udp"] += 1
            proto = "UDP"
        elif packet.haslayer(ARP):
            self.stats["arp"] += 1
            proto = "ARP"
        else:
            self.stats["other"] += 1
            proto = "OTHER"

        entry = {"time": now, "proto": proto, "len": len(packet)}
        if packet.haslayer(IP):
            entry["src"] = packet[IP].src
            entry["dst"] = packet[IP].dst
            self.stats["src_ips"][packet[IP].src] += 1
            self.stats["dst_ips"][packet[IP].dst] += 1
        if packet.haslayer(TCP):
            entry["sport"] = packet[TCP].sport
            entry["dport"] = packet[TCP].dport
            self.stats["ports"][packet[TCP].dport] += 1
        elif packet.haslayer(UDP):
            entry["sport"] = packet[UDP].sport
            entry["dport"] = packet[UDP].dport
            self.stats["ports"][packet[UDP].dport] += 1

        self.stats["packet_history"].append(entry)
        if len(self.stats["packet_history"]) > 500:
            self.stats["packet_history"] = self.stats["packet_history"][-500:]

        if self.callback:
            self.callback(entry)

    def start(self, count=0, timeout=None):
        if not SCAPY_AVAIL:
            print("[CAPTURE] Scapy not available. Install with: pip install scapy")
            return False
        if self.running:
            return False

        self.running = True
        self.stats["start_time"] = datetime.now()

        def _sniff():
            sniff(iface=self.interface, prn=self._handle_packet,
                  store=False, count=count, timeout=timeout)

        self.thread = threading.Thread(target=_sniff, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def get_rate(self, window_sec=5):
        if not self.stats["packet_history"]:
            return 0
        cutoff = datetime.now().timestamp() - window_sec
        recent = [p for p in self.stats["packet_history"]
                  if p["time"].timestamp() > cutoff]
        return len(recent) / window_sec if recent else 0

    def summary(self):
        s = self.stats
        elapsed = (datetime.now() - s["start_time"]).total_seconds() if s["start_time"] else 0
        rate = s["total_packets"] / elapsed if elapsed > 0 else 0
        top_ips = sorted(s["src_ips"].items(), key=lambda x: x[1], reverse=True)[:5]
        top_ports = sorted(s["ports"].items(), key=lambda x: x[1], reverse=True)[:5]
        return {
            "total": s["total_packets"],
            "elapsed": round(elapsed, 1),
            "rate": round(rate, 1),
            "tcp": s["tcp"], "udp": s["udp"], "arp": s["arp"],
            "top_src_ips": top_ips,
            "top_ports": top_ports,
        }
