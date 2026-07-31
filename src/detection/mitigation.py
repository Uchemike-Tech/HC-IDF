import os
import subprocess
import csv
import time
import threading
from datetime import datetime
from pathlib import Path
from collections import defaultdict


class IPBlocker:
    def __init__(self, config: dict):
        self.config = config
        self.blocked_ips = {}
        self.persist = config.get("persist_blocks", True)

    def block_ip(self, ip: str, reason: str = "", duration: int = None) -> bool:
        if ip in self.blocked_ips:
            return False
        if os.name == "nt":
            cmd = f'netsh advfirewall firewall add rule name="HC-IDF Block {ip}" dir=in interface=any action=block remoteip={ip}'
        else:
            cmd = f"iptables -A INPUT -s {ip} -j DROP"
        try:
            subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
            expiry = time.time() + (duration or self.config.get("default_block_duration", 3600))
            self.blocked_ips[ip] = {
                "ip": ip,
                "reason": reason,
                "blocked_at": datetime.now().isoformat(),
                "expires_at": datetime.fromtimestamp(expiry).isoformat(),
                "duration": duration or self.config.get("default_block_duration", 3600),
            }
            return True
        except Exception:
            return False

    def unblock_ip(self, ip: str) -> bool:
        if ip not in self.blocked_ips:
            return False
        if os.name == "nt":
            cmd = f'netsh advfirewall firewall delete rule name="HC-IDF Block {ip}"'
        else:
            cmd = f"iptables -D INPUT -s {ip} -j DROP"
        try:
            subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
            del self.blocked_ips[ip]
            return True
        except Exception:
            return False

    def get_blocked(self) -> list:
        now = time.time()
        expired = [ip for ip, info in self.blocked_ips.items()
                   if datetime.fromisoformat(info["expires_at"]).timestamp() < now]
        for ip in expired:
            self.unblock_ip(ip)
        return list(self.blocked_ips.values())

    def cleanup_expired(self):
        self.get_blocked()


class SessionTerminator:
    def __init__(self, config: dict):
        self.config = config
        self.terminated_sessions = []

    def terminate_session(self, src_ip: str, dst_ip: str, src_port: int = None, dst_port: int = None) -> bool:
        if os.name == "nt":
            if dst_port:
                cmd = f"netstat -ano | findstr :{dst_port}"
            else:
                cmd = f"netstat -ano | findstr {src_ip}"
        else:
            if dst_port:
                cmd = f"ss -K dst {dst_ip} dport = {dst_port}"
            else:
                cmd = f"tcpkill -i any host {src_ip}"

        record = {
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "terminated_at": datetime.now().isoformat(),
        }
        self.terminated_sessions.append(record)
        return True

    def get_terminated_sessions(self) -> list:
        return self.terminated_sessions[-50:]


class MitigationOrchestrator:
    def __init__(self, config: dict):
        self.config = config.get("mitigation", {})
        self.ip_blocker = IPBlocker(self.config)
        self.session_terminator = SessionTerminator(self.config)
        self.mitigation_log = []
        self.auto_mitigation = self.config.get("auto_mitigation", False)
        self.mitigation_count = 0
        self._lock = threading.Lock()
        self.log_path = Path("data/mitigation_log.csv")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def handle_mitm_alert(self, alert: dict) -> dict:
        if not alert.get("mitm_alert"):
            return {"action": "none", "reason": "No MITM alert"}

        mitm_types = alert.get("mitm_type", []) or []
        src_ip = alert.get("src_ip", "")
        dst_ip = alert.get("dst_ip", "")
        severity = self._assess_severity(mitm_types)

        actions = []
        if self.auto_mitigation or severity == "critical":
            if src_ip:
                blocked = self.ip_blocker.block_ip(
                    src_ip,
                    reason=f"MITM: {', '.join(mitm_types)}",
                    duration=self.config.get("auto_block_duration", 1800),
                )
                if blocked:
                    actions.append({"action": "block_ip", "target": src_ip, "severity": severity})

            terminated = self.session_terminator.terminate_session(src_ip, dst_ip)
            if terminated:
                actions.append({"action": "terminate_session", "target": f"{src_ip}→{dst_ip}", "severity": severity})

        result = {
            "timestamp": datetime.now().isoformat(),
            "mitm_types": mitm_types,
            "severity": severity,
            "actions": actions,
            "auto_mitigated": self.auto_mitigation,
            "action_count": len(actions),
        }

        with self._lock:
            self.mitigation_log.append(result)
            self.mitigation_count += 1
            self._append_log(result)

        return result

    def _assess_severity(self, mitm_types: list) -> str:
        critical_types = {"SESSION_HIJACKING", "DNS_POISONING"}
        for t in mitm_types:
            if t in critical_types:
                return "critical"
        return "high" if mitm_types else "none"

    def _append_log(self, entry: dict):
        filepath = self.log_path
        exists = filepath.exists()
        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["timestamp", "mitm_types", "severity", "actions", "auto_mitigated", "action_count"])
            writer.writerow([
                entry["timestamp"],
                ";".join(entry["mitm_types"]),
                entry["severity"],
                ";".join([f"{a['action']}:{a['target'].replace('\u2192', '->')}" for a in entry["actions"]]),
                entry["auto_mitigated"],
                entry["action_count"],
            ])

    def get_stats(self) -> dict:
        with self._lock:
            active_blocks = self.ip_blocker.get_blocked()
            return {
                "total_mitigations": self.mitigation_count,
                "active_blocks": len(active_blocks),
                "blocked_ips": active_blocks,
                "recent_actions": self.mitigation_log[-20:] if self.mitigation_log else [],
                "auto_mitigation_enabled": self.auto_mitigation,
                "terminated_sessions": self.session_terminator.get_terminated_sessions(),
            }

    def toggle_auto_mitigation(self, enabled: bool = None):
        if enabled is not None:
            self.auto_mitigation = enabled
        else:
            self.auto_mitigation = not self.auto_mitigation
        return self.auto_mitigation

    def unblock_ip(self, ip: str) -> bool:
        return self.ip_blocker.unblock_ip(ip)

    def get_logs(self, limit: int = 50) -> list:
        if not self.log_path.exists():
            return []
        with open(self.log_path, "r") as f:
            reader = csv.DictReader(f)
            return list(reader)[-limit:]
