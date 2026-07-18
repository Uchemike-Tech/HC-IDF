import threading
import time
import numpy as np
from collections import defaultdict
from datetime import datetime

try:
    from scapy.all import sniff, IP, TCP, UDP, ARP, DNS, Ether, conf
    SCAPY_AVAIL = True
except ImportError:
    SCAPY_AVAIL = False

COMMON_PORTS = {22, 25, 53, 80, 110, 123, 143, 443, 445, 993, 995, 3306, 3389, 5432, 8080, 8443}
UNUSUAL_PORTS = {7, 9, 19, 21, 23, 135, 137, 139, 389, 502, 2323, 3128, 4444, 6660, 6661, 6662, 6663, 6664, 6665, 6666, 6667, 6668, 6669, 12345, 31337, 44444, 65535}

class LiveCapture:
    def __init__(self, interface=None):
        self.interface = interface
        self.running = False
        self.thread = None
        self.callback = None
        self.stats = {
            "total_packets": 0,
            "tcp": 0, "udp": 0, "arp": 0, "dns": 0, "other": 0,
            "src_ips": defaultdict(int),
            "dst_ips": defaultdict(int),
            "ports": defaultdict(int),
            "start_time": None,
            "packet_history": [],
        }
        self.flows = {}
        self.ip_ports = defaultdict(set)
        self.alerts = []
        self.alert_id = 0

    def _detect(self, src, dst, dport, proto):
        alerts = []
        is_unusual = dport in UNUSUAL_PORTS or (dport < 1024 and dport not in COMMON_PORTS)
        if is_unusual:
            self.alert_id += 1
            alerts.append({
                "id": self.alert_id, "type": "Unusual Port",
                "severity": "Medium", "src": src, "dst": dst,
                "detail": f"Traffic on unusual port {dport}/{proto}",
                "time": datetime.now().isoformat()
            })
        ip_key = src
        self.ip_ports[ip_key].add(dport)
        if len(self.ip_ports[ip_key]) >= 10 and len(self.ip_ports[ip_key]) % 5 == 0:
            self.alert_id += 1
            alerts.append({
                "id": self.alert_id, "type": "Port Scan",
                "severity": "High", "src": src, "dst": dst,
                "detail": f"{src} contacted {len(self.ip_ports[ip_key])} unique ports — possible scan",
                "time": datetime.now().isoformat()
            })
        if proto == "ARP":
            self.alert_id += 1
            alerts.append({
                "id": self.alert_id, "type": "ARP Activity",
                "severity": "Info", "src": src, "dst": dst,
                "detail": "ARP packet detected — possible MITM reconnaissance",
                "time": datetime.now().isoformat()
            })
        return alerts

    def _handle_packet(self, packet):
        if not self.running:
            return
        self.stats["total_packets"] += 1
        now = datetime.now()

        if packet.haslayer(TCP):
            self.stats["tcp"] += 1; proto = "TCP"
        elif packet.haslayer(UDP):
            self.stats["udp"] += 1; proto = "UDP"
        elif packet.haslayer(ARP):
            self.stats["arp"] += 1; proto = "ARP"
        else:
            self.stats["other"] += 1; proto = "OTHER"

        entry = {"time": now, "proto": proto, "len": len(packet)}
        src_ip = dst_ip = sport = dport = None
        if packet.haslayer(IP):
            src_ip = packet[IP].src; dst_ip = packet[IP].dst
            entry["src"] = src_ip; entry["dst"] = dst_ip
            self.stats["src_ips"][src_ip] += 1
            self.stats["dst_ips"][dst_ip] += 1
        if packet.haslayer(TCP):
            sport = packet[TCP].sport; dport = packet[TCP].dport
            entry["sport"] = sport; entry["dport"] = dport
            self.stats["ports"][dport] += 1
        elif packet.haslayer(UDP):
            sport = packet[UDP].sport; dport = packet[UDP].dport
            entry["sport"] = sport; entry["dport"] = dport
            self.stats["ports"][dport] += 1

        self.stats["packet_history"].append(entry)
        if len(self.stats["packet_history"]) > 500:
            self.stats["packet_history"] = self.stats["packet_history"][-500:]

        if src_ip and dport:
            new_alerts = self._detect(src_ip, dst_ip, dport, proto)
            self.alerts.extend(new_alerts)
            if len(self.alerts) > 200:
                self.alerts = self.alerts[-200:]

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
            sock = conf.L3socket(iface=self.interface)
            sniff(opened_socket=sock, prn=self._handle_packet,
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
        recent = [p for p in self.stats["packet_history"] if p["time"].timestamp() > cutoff]
        return len(recent) / window_sec if recent else 0

    def summary(self):
        s = self.stats
        elapsed = (datetime.now() - s["start_time"]).total_seconds() if s["start_time"] else 0
        rate = s["total_packets"] / elapsed if elapsed > 0 else 0
        top_ips = sorted(s["src_ips"].items(), key=lambda x: x[1], reverse=True)[:5]
        top_ports = sorted(s["ports"].items(), key=lambda x: x[1], reverse=True)[:5]
        recent_alerts = self.alerts[-10:] if self.alerts else []
        return {
            "total": s["total_packets"],
            "elapsed": round(elapsed, 1),
            "rate": round(rate, 1),
            "tcp": s["tcp"], "udp": s["udp"], "arp": s["arp"],
            "top_src_ips": top_ips,
            "top_ports": top_ports,
            "alert_count": len(self.alerts),
            "recent_alerts": [
                {"id": a["id"], "type": a["type"], "severity": a["severity"],
                 "src": a["src"], "detail": a["detail"]}
                for a in recent_alerts
            ],
        }
