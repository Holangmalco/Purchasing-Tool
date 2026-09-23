"""SQLite-backed Vendor Master Cache module for reliable supplier identification and OCR fallback."""
from __future__ import annotations

import datetime
import json
import os
import re
import sqlite3
from pathlib import Path
import contextlib
from typing import Any, Iterator, Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor_master.db")


@contextlib.contextmanager
def _get_connection(db_path: Optional[str] = None) -> Iterator[sqlite3.Connection]:
    target_path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_vendor_db(db_path: Optional[str] = None) -> None:
    """Initializes the vendor master SQLite table and indexes."""
    with _get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vendors (
                business_number TEXT PRIMARY KEY,
                vendor_name TEXT NOT NULL,
                aliases TEXT NOT NULL DEFAULT '[]',
                representative TEXT NOT NULL DEFAULT '',
                accounts TEXT NOT NULL DEFAULT '[]',
                phone TEXT NOT NULL DEFAULT '',
                email TEXT NOT NULL DEFAULT '',
                verified_count INTEGER NOT NULL DEFAULT 1,
                last_seen TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vendors_name ON vendors (vendor_name)")
        conn.commit()


def normalize_biz_num(value: str) -> str:
    """Normalizes business number into 10-digit XXX-XX-XXXXX format."""
    digits = re.sub(r"\D", "", str(value))
    if len(digits) == 10:
        return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
    return digits


def get_vendor_by_biz_num(biz_num: str, db_path: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Retrieves vendor master record by normalized business number."""
    clean_bn = normalize_biz_num(biz_num)
    if not clean_bn or len(clean_bn) != 12:
        return None
    init_vendor_db(db_path)
    with _get_connection(db_path) as conn:
        cursor = conn.execute("SELECT * FROM vendors WHERE business_number = ?", (clean_bn,))
        row = cursor.fetchone()
        if not row:
            return None
        data = dict(row)
        data["aliases"] = json.loads(data["aliases"])
        data["accounts"] = json.loads(data["accounts"])
        return data


def get_vendor_by_name(vendor_name: str, db_path: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Retrieves vendor master record by exact or alias vendor name."""
    clean_name = re.sub(r"\(주\)|\(유\)|주식회사|유한회사|\s+", "", str(vendor_name)).lower()
    if not clean_name:
        return None
    init_vendor_db(db_path)
    with _get_connection(db_path) as conn:
        cursor = conn.execute("SELECT * FROM vendors")
        for row in cursor.fetchall():
            canonical = re.sub(r"\(주\)|\(유\)|주식회사|유한회사|\s+", "", row["vendor_name"]).lower()
            if canonical == clean_name:
                data = dict(row)
                data["aliases"] = json.loads(data["aliases"])
                data["accounts"] = json.loads(data["accounts"])
                return data
            aliases = json.loads(row["aliases"])
            for alias in aliases:
                norm_alias = re.sub(r"\(주\)|\(유\)|주식회사|유한회사|\s+", "", alias).lower()
                if norm_alias == clean_name:
                    data = dict(row)
                    data["aliases"] = aliases
                    data["accounts"] = json.loads(data["accounts"])
                    return data
    return None


def save_vendor(
    biz_num: str,
    vendor_name: str,
    representative: str = "",
    bank: str = "",
    account_num: str = "",
    account_holder: str = "",
    phone: str = "",
    email: str = "",
    db_path: Optional[str] = None,
) -> bool:
    """Saves or safely updates vendor master record in SQLite."""
    clean_bn = normalize_biz_num(biz_num)
    if not clean_bn or len(clean_bn) != 12:
        return False
    clean_vname = vendor_name.strip()
    if not clean_vname:
        return False

    init_vendor_db(db_path)
    now_str = datetime.date.today().isoformat()

    with _get_connection(db_path) as conn:
        cursor = conn.execute("SELECT * FROM vendors WHERE business_number = ?", (clean_bn,))
        existing = cursor.fetchone()

        if existing:
            current_aliases = json.loads(existing["aliases"])
            if clean_vname != existing["vendor_name"] and clean_vname not in current_aliases:
                current_aliases.append(clean_vname)

            current_accounts = json.loads(existing["accounts"])
            if account_num:
                clean_acc = re.sub(r"[^\d-]", "", account_num).strip()
                if clean_acc:
                    acc_exists = any(a.get("account") == clean_acc for a in current_accounts)
                    if not acc_exists:
                        current_accounts.append({
                            "bank": bank.strip(),
                            "account": clean_acc,
                            "holder": account_holder.strip() or clean_vname,
                            "added_at": now_str,
                        })

            rep_val = representative.strip() or existing["representative"]
            phone_val = phone.strip() or existing["phone"]
            email_val = email.strip() or existing["email"]
            new_count = existing["verified_count"] + 1

            conn.execute(
                """
                UPDATE vendors
                SET aliases = ?, representative = ?, accounts = ?, phone = ?, email = ?,
                    verified_count = ?, last_seen = ?
                WHERE business_number = ?
                """,
                (
                    json.dumps(current_aliases, ensure_ascii=False),
                    rep_val,
                    json.dumps(current_accounts, ensure_ascii=False),
                    phone_val,
                    email_val,
                    new_count,
                    now_str,
                    clean_bn,
                ),
            )
        else:
            aliases = [clean_vname]
            accounts = []
            if account_num:
                clean_acc = re.sub(r"[^\d-]", "", account_num).strip()
                if clean_acc:
                    accounts.append({
                        "bank": bank.strip(),
                        "account": clean_acc,
                        "holder": account_holder.strip() or clean_vname,
                        "added_at": now_str,
                    })

            conn.execute(
                """
                INSERT INTO vendors (
                    business_number, vendor_name, aliases, representative,
                    accounts, phone, email, verified_count, last_seen
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    clean_bn,
                    clean_vname,
                    json.dumps(aliases, ensure_ascii=False),
                    representative.strip(),
                    json.dumps(accounts, ensure_ascii=False),
                    phone.strip(),
                    email.strip(),
                    now_str,
                ),
            )
        conn.commit()
    return True


def get_all_vendors(db_path: Optional[str] = None) -> list[dict[str, Any]]:
    """Retrieves all vendor records sorted by most recently verified."""
    init_vendor_db(db_path)
    with _get_connection(db_path) as conn:
        cursor = conn.execute("SELECT * FROM vendors ORDER BY last_seen DESC, verified_count DESC")
        results = []
        for row in cursor.fetchall():
            data = dict(row)
            data["aliases"] = json.loads(data["aliases"])
            data["accounts"] = json.loads(data["accounts"])
            results.append(data)
        return results


def delete_vendor(biz_num: str, db_path: Optional[str] = None) -> bool:
    """Deletes a vendor record by business number."""
    clean_bn = normalize_biz_num(biz_num)
    init_vendor_db(db_path)
    with _get_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM vendors WHERE business_number = ?", (clean_bn,))
        conn.commit()
        return cursor.rowcount > 0


def auto_cache_verified_documents(documents: list[Any], db_path: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Inspects verified purchasing documents and caches the vendor into master DB."""
    from purchase_verifier.core import DocumentType, is_valid_business_number

    # 1. Look for business license or lowest quote with valid business number
    target_bn = ""
    target_vname = ""
    target_rep = ""
    target_bank = ""
    target_acc_num = ""
    target_acc_holder = ""
    target_phone = ""
    target_email = ""

    # Business License
    for d in documents:
        if d.document_type == DocumentType.BUSINESS_LICENSE:
            if d.business_number.raw and is_valid_business_number(d.business_number.raw):
                target_bn = d.business_number.raw
            if d.vendor_name.raw:
                target_vname = d.vendor_name.raw
            if d.representative.raw:
                target_rep = d.representative.raw

    # Bank copy
    for d in documents:
        if d.document_type == DocumentType.BANK_COPY:
            if d.account_number.raw:
                target_acc_num = d.account_number.raw
            if d.account_holder.raw:
                target_acc_holder = d.account_holder.raw

    # Quote
    for d in documents:
        if d.document_type == DocumentType.QUOTE:
            if not target_bn and d.business_number.raw and is_valid_business_number(d.business_number.raw):
                target_bn = d.business_number.raw
            if not target_vname and d.vendor_name.raw:
                target_vname = d.vendor_name.raw
            if not target_rep and d.representative.raw:
                target_rep = d.representative.raw
            if d.phone.raw and not target_phone:
                target_phone = d.phone.raw
            if d.email.raw and not target_email:
                target_email = d.email.raw

    if target_bn and target_vname:
        ok = save_vendor(
            biz_num=target_bn,
            vendor_name=target_vname,
            representative=target_rep,
            bank=target_bank,
            account_num=target_acc_num,
            account_holder=target_acc_holder,
            phone=target_phone,
            email=target_email,
            db_path=db_path,
        )
        if ok:
            return {
                "business_number": target_bn,
                "vendor_name": target_vname,
                "representative": target_rep,
                "account_number": target_acc_num,
            }
    return None
