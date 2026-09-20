"""Dataset feature schemas, label conventions, and column aliases.

CIC-IDS2017 / CSE-CIC-IDS2018 use CICFlowMeter-style flow features. UNSW-NB15
uses a different 42-column machine-learning export. CIC-IoT-2023 is wired as a
stretch schema with a synthetic stand-in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CICIDS2017_FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean",
    "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

# CSE-CIC-IDS2018 AWS CSVs use abbreviated CICFlowMeter names.
CICIDS2018_FEATURES = [
    "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max", "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt",
    "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt", "Down/Up Ratio",
    "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
    "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts",
    "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

# Numeric columns from UNSW_NB15_training-set.csv after dropping id / categoricals.
UNSW_NB15_NUMERIC = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports",
]

UNSW_NB15_CATEGORICAL = ["proto", "service", "state"]

# Compact CIC-IoT-2023-style numeric stand-in (official CSVs have 40+ flow stats).
CICIOT2023_FEATURES = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate",
    "Srate", "Drate", "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP",
    "ARP", "ICMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG", "Std",
    "Tot size", "IAT", "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight",
]

# Features an attacker cannot rewrite without breaking protocol semantics
# (ports, flags, protocol identifiers). Timing/size/rate features remain open.
CIC_FROZEN = {
    "Destination Port", "Dst Port", "Protocol",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
    "ACK Flag Cnt", "URG Flag Cnt", "ECE Flag Cnt",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
}

UNSW_FROZEN = {
    "sttl", "dttl", "swin", "dwin", "is_ftp_login", "is_sm_ips_ports",
    "proto", "service", "state",
}

CICIOT_FROZEN = {
    "Protocol Type", "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP",
    "DHCP", "ARP", "ICMP", "IPv", "LLC",
}

# Map abbreviated CIC-IDS2018 names onto the CIC-IDS2017 MachineLearningCSV names
# so the two CICFlowMeter families can share a feature space after alignment.
CIC2018_TO_CIC2017 = {
    "Dst Port": "Destination Port",
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "Fwd IAT Tot": "Fwd IAT Total",
    "Bwd IAT Tot": "Bwd IAT Total",
    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s",
    "Bwd Pkts/s": "Bwd Packets/s",
    "Pkt Len Min": "Min Packet Length",
    "Pkt Len Max": "Max Packet Length",
    "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",
    "FIN Flag Cnt": "FIN Flag Count",
    "SYN Flag Cnt": "SYN Flag Count",
    "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count",
    "ACK Flag Cnt": "ACK Flag Count",
    "URG Flag Cnt": "URG Flag Count",
    "ECE Flag Cnt": "ECE Flag Count",
    "Pkt Size Avg": "Average Packet Size",
    "Fwd Seg Size Avg": "Avg Fwd Segment Size",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size",
    "Fwd Byts/b Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Pkts/b Avg": "Fwd Avg Packets/Bulk",
    "Fwd Blk Rate Avg": "Fwd Avg Bulk Rate",
    "Bwd Byts/b Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Pkts/b Avg": "Bwd Avg Packets/Bulk",
    "Bwd Blk Rate Avg": "Bwd Avg Bulk Rate",
    "Subflow Fwd Pkts": "Subflow Fwd Packets",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Subflow Bwd Byts": "Subflow Bwd Bytes",
    "Init Fwd Win Byts": "Init_Win_bytes_forward",
    "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
}

BENIGN_TOKENS = {
    "BENIGN", "Benign", "benign", "NORMAL", "Normal", "normal",
    "0", 0,
}

CIC_ATTACK_NAMES = [
    "DoS Hulk", "DDoS", "PortScan", "FTP-Patator", "SSH-Patator",
    "Bot", "Infiltration", "DoS GoldenEye", "DoS Slowloris",
]

UNSW_ATTACK_NAMES = [
    "Fuzzers", "Analysis", "Backdoor", "DoS", "Exploits",
    "Generic", "Reconnaissance", "Shellcode", "Worms",
]

CICIOT_ATTACK_NAMES = [
    "DDoS-ICMP_Flood", "DDoS-SYN_Flood", "DDoS-TCP_Flood",
    "DoS-TCP_Flood", "Mirai-greeth_flood", "Recon-PortScan",
    "SQL_Injection", "Uploading_Attack",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    family: str
    features: list[str]
    frozen: set[str]
    label_column: str = "Label"
    multiclass_column: str | None = None
    drop_columns: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()
    notes: str = ""
    aliases: dict[str, str] = field(default_factory=dict)


DATASET_SPECS: dict[str, DatasetSpec] = {
    "cicids2017": DatasetSpec(
        name="cicids2017",
        family="cicflowmeter",
        features=list(CICIDS2017_FEATURES),
        frozen=set(CIC_FROZEN),
        label_column="Label",
        notes="CIC-IDS2017 MachineLearningCSV / CICFlowMeter features.",
    ),
    "cicids2018": DatasetSpec(
        name="cicids2018",
        family="cicflowmeter",
        features=list(CICIDS2018_FEATURES),
        frozen=set(CIC_FROZEN),
        label_column="Label",
        drop_columns=("Timestamp",),
        notes="CSE-CIC-IDS2018 AWS CSV subset (per-day files).",
        aliases=dict(CIC2018_TO_CIC2017),
    ),
    "unsw_nb15": DatasetSpec(
        name="unsw_nb15",
        family="unsw",
        features=list(UNSW_NB15_NUMERIC),
        frozen=set(UNSW_FROZEN),
        label_column="label",
        multiclass_column="attack_cat",
        drop_columns=("id",),
        categorical=tuple(UNSW_NB15_CATEGORICAL),
        notes="UNSW-NB15 official training/testing CSV export.",
    ),
    "ciciot2023": DatasetSpec(
        name="ciciot2023",
        family="ciciot",
        features=list(CICIOT2023_FEATURES),
        frozen=set(CICIOT_FROZEN),
        label_column="label",
        notes="Optional CIC-IoT-2023 stretch schema (synthetic stand-in supported).",
    ),
}


def normalize_dataset_name(name: str | None) -> str:
    if not name:
        return "cicids2017"
    key = name.strip().lower().replace("-", "_").replace(" ", "")
    aliases = {
        "cic_ids2017": "cicids2017",
        "cicids_2017": "cicids2017",
        "cse_cic_ids2018": "cicids2018",
        "cse-cic-ids2018": "cicids2018",
        "ids2018": "cicids2018",
        "unsw-nb15": "unsw_nb15",
        "unsw": "unsw_nb15",
        "cic_iot_2023": "ciciot2023",
        "cic-iot-2023": "ciciot2023",
        "synthetic": "cicids2017",
        "synthetic_cicids2017": "cicids2017",
    }
    return aliases.get(key, key)


# Re-export used by the original preprocessing module.
FROZEN_FEATURES_COMPAT = {
    "Destination Port", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
    "ECE Flag Count", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
}


def get_spec(name: str | None) -> DatasetSpec:
    key = normalize_dataset_name(name)
    if key not in DATASET_SPECS:
        known = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unknown dataset '{name}'. Known: {known}")
    return DATASET_SPECS[key]
