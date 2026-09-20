#!/usr/bin/env python3
"""Regenerate tiny official-schema CSV fixtures (headers match published dumps)."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "tests" / "fixtures"

# Exact MachineLearningCSV header (leading spaces + duplicate Fwd Header Length).
CIC2017_HEADER = [
    " Destination Port",
    " Flow Duration",
    " Total Fwd Packets",
    " Total Backward Packets",
    "Total Length of Fwd Packets",
    " Total Length of Bwd Packets",
    " Fwd Packet Length Max",
    " Fwd Packet Length Min",
    " Fwd Packet Length Mean",
    " Fwd Packet Length Std",
    "Bwd Packet Length Max",
    " Bwd Packet Length Min",
    " Bwd Packet Length Mean",
    " Bwd Packet Length Std",
    "Flow Bytes/s",
    " Flow Packets/s",
    " Flow IAT Mean",
    " Flow IAT Std",
    " Flow IAT Max",
    " Flow IAT Min",
    "Fwd IAT Total",
    " Fwd IAT Mean",
    " Fwd IAT Std",
    " Fwd IAT Max",
    " Fwd IAT Min",
    "Bwd IAT Total",
    " Bwd IAT Mean",
    " Bwd IAT Std",
    " Bwd IAT Max",
    " Bwd IAT Min",
    "Fwd PSH Flags",
    " Bwd PSH Flags",
    " Fwd URG Flags",
    " Bwd URG Flags",
    " Fwd Header Length",
    " Bwd Header Length",
    "Fwd Packets/s",
    " Bwd Packets/s",
    " Min Packet Length",
    " Max Packet Length",
    " Packet Length Mean",
    " Packet Length Std",
    " Packet Length Variance",
    "FIN Flag Count",
    " SYN Flag Count",
    " RST Flag Count",
    " PSH Flag Count",
    " ACK Flag Count",
    " URG Flag Count",
    " CWE Flag Count",
    " ECE Flag Count",
    " Down/Up Ratio",
    " Average Packet Size",
    " Avg Fwd Segment Size",
    " Avg Bwd Segment Size",
    " Fwd Header Length",
    "Fwd Avg Bytes/Bulk",
    " Fwd Avg Packets/Bulk",
    " Fwd Avg Bulk Rate",
    " Bwd Avg Bytes/Bulk",
    " Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    " Subflow Fwd Bytes",
    " Subflow Bwd Packets",
    " Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    " Init_Win_bytes_backward",
    " act_data_pkt_fwd",
    " min_seg_size_forward",
    "Active Mean",
    " Active Std",
    " Active Max",
    " Active Min",
    "Idle Mean",
    " Idle Std",
    " Idle Max",
    " Idle Min",
    " Label",
]

# CSE-CIC-IDS2018 AWS CSV header (abbreviated CICFlowMeter + Timestamp).
CIC2018_HEADER = [
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Fwd Pkt Len Max",
    "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean",
    "Fwd Pkt Len Std",
    "Bwd Pkt Len Max",
    "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean",
    "Bwd Pkt Len Std",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Tot",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Tot",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Len",
    "Bwd Header Len",
    "Fwd Pkts/s",
    "Bwd Pkts/s",
    "Pkt Len Min",
    "Pkt Len Max",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Pkt Len Var",
    "FIN Flag Cnt",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "URG Flag Cnt",
    "CWE Flag Count",
    "ECE Flag Cnt",
    "Down/Up Ratio",
    "Pkt Size Avg",
    "Fwd Seg Size Avg",
    "Bwd Seg Size Avg",
    "Fwd Byts/b Avg",
    "Fwd Pkts/b Avg",
    "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg",
    "Bwd Pkts/b Avg",
    "Bwd Blk Rate Avg",
    "Subflow Fwd Pkts",
    "Subflow Fwd Byts",
    "Subflow Bwd Pkts",
    "Subflow Bwd Byts",
    "Init Fwd Win Byts",
    "Init Bwd Win Byts",
    "Fwd Act Data Pkts",
    "Fwd Seg Size Min",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    "Label",
]

# Official UNSW_NB15_training-set.csv column order.
UNSW_HEADER = [
    "id",
    "dur",
    "proto",
    "service",
    "state",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
    "attack_cat",
    "label",
]


def _cic_row(port: int, duration: int, label: str, *, attack: bool) -> list:
    n = len(CIC2017_HEADER)
    row = [0] * n
    row[0] = port
    row[1] = duration
    row[2] = 12 if attack else 4
    row[3] = 8 if attack else 3
    row[4] = 800 if attack else 200
    row[5] = 400 if attack else 150
    row[14] = 1.2e5 if attack else 2.5e4
    row[15] = 80 if attack else 12
    row[43] = 0
    row[44] = 1 if attack else 0
    row[47] = 1
    row[-1] = label
    return row


def _cic2018_row(port: int, proto: int, ts: str, label: str, *, attack: bool) -> list:
    n = len(CIC2018_HEADER)
    row = [0] * n
    row[0] = port
    row[1] = proto
    row[2] = ts
    row[3] = 90000 if attack else 12000
    row[4] = 20 if attack else 5
    row[5] = 10 if attack else 3
    row[16] = 9.5e4 if attack else 1.8e4
    row[-1] = label
    return row


def _unsw_row(i: int, *, attack: bool, cat: str) -> list:
    if attack:
        return [
            i, 0.42, "tcp", "http", "FIN", 24, 18, 1800, 900, 2100.0,
            254, 252, 4.1e4, 2.0e4, 2, 1, 12.0, 8.0, 3.1, 1.4,
            255, 1.1e9, 2.2e9, 255, 0.04, 0.02, 0.02, 75, 50,
            1, 120, 6, 2, 4, 3, 2, 5, 0, 0, 1, 4, 5, 0, cat, 1,
        ]
    return [
        i, 0.08, "tcp", "dns", "CON", 4, 4, 280, 260, 180.0,
        31, 29, 3.5e3, 3.2e3, 0, 0, 40.0, 38.0, 0.2, 0.1,
        255, 4.0e8, 3.5e8, 255, 0.01, 0.005, 0.005, 70, 65,
        0, 0, 2, 0, 1, 1, 1, 1, 0, 0, 0, 2, 2, 0, cat, 0,
    ]


def _write(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    cic2017 = [
        _cic_row(80, 15000, "BENIGN", attack=False),
        _cic_row(443, 22000, "BENIGN", attack=False),
        _cic_row(53, 800, "BENIGN", attack=False),
        _cic_row(22, 41000, "BENIGN", attack=False),
        _cic_row(80, 9000, "BENIGN", attack=False),
        _cic_row(123, 600, "BENIGN", attack=False),
        _cic_row(443, 18000, "BENIGN", attack=False),
        _cic_row(25, 12000, "BENIGN", attack=False),
        _cic_row(80, 190000, "DoS Hulk", attack=True),
        _cic_row(80, 210000, "DDoS", attack=True),
        _cic_row(21, 88000, "FTP-Patator", attack=True),
        _cic_row(22, 76000, "SSH-Patator", attack=True),
    ]
    _write(DEST / "cicids2017_official_sample.csv", CIC2017_HEADER, cic2017)

    cic2018 = [
        _cic2018_row(80, 6, "14/02/2018 08:01:00", "Benign", attack=False),
        _cic2018_row(443, 6, "14/02/2018 08:01:05", "Benign", attack=False),
        _cic2018_row(53, 17, "14/02/2018 08:01:10", "Benign", attack=False),
        _cic2018_row(22, 6, "14/02/2018 08:01:15", "Benign", attack=False),
        _cic2018_row(80, 6, "14/02/2018 08:01:20", "Benign", attack=False),
        _cic2018_row(443, 6, "14/02/2018 08:01:25", "Benign", attack=False),
        _cic2018_row(123, 17, "14/02/2018 08:01:30", "Benign", attack=False),
        _cic2018_row(25, 6, "14/02/2018 08:01:35", "Benign", attack=False),
        _cic2018_row(21, 6, "14/02/2018 10:12:00", "FTP-BruteForce", attack=True),
        _cic2018_row(22, 6, "14/02/2018 10:15:00", "SSH-Bruteforce", attack=True),
        _cic2018_row(80, 6, "15/02/2018 11:00:00", "DoS-GoldenEye", attack=True),
        _cic2018_row(80, 6, "16/02/2018 13:00:00", "DoS-Hulk", attack=True),
    ]
    _write(DEST / "cicids2018_official_sample.csv", CIC2018_HEADER, cic2018)

    unsw = [
        _unsw_row(1, attack=False, cat="Normal"),
        _unsw_row(2, attack=False, cat="Normal"),
        _unsw_row(3, attack=False, cat="Normal"),
        _unsw_row(4, attack=False, cat="Normal"),
        _unsw_row(5, attack=False, cat="Normal"),
        _unsw_row(6, attack=False, cat="Normal"),
        _unsw_row(7, attack=False, cat="Normal"),
        _unsw_row(8, attack=False, cat="Normal"),
        _unsw_row(9, attack=True, cat="Exploits"),
        _unsw_row(10, attack=True, cat="DoS"),
        _unsw_row(11, attack=True, cat="Generic"),
        _unsw_row(12, attack=True, cat="Reconnaissance"),
    ]
    _write(DEST / "unsw_nb15_official_sample.csv", UNSW_HEADER, unsw)
    print(f"Wrote fixtures under {DEST}")


if __name__ == "__main__":
    main()
