import base64
import datetime
import json
from typing import Any, Dict, List, Optional, Tuple


from config import MYSQL_CONFIG


class _DBBackend:
    """Lightweight adapter to support either PyMySQL or mysql-connector-python.

    Prefers PyMySQL if available, falls back to mysql-connector-python.
    """

    def __init__(self) -> None:
        self._driver = None
        self._use_pymysql = False
        try:
            import pymysql  # type: ignore

            self._driver = pymysql
            self._use_pymysql = True
        except Exception:
            try:
                import mysql.connector  # type: ignore

                self._driver = mysql.connector
                self._use_pymysql = False
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "No MySQL driver found. Please install 'pymysql' or 'mysql-connector-python'."
                ) from exc

    def connect(self):  # type: ignore[no-untyped-def]
        if self._use_pymysql:
            return self._driver.connect(
                host=MYSQL_CONFIG['host'],
                port=MYSQL_CONFIG['port'],
                user=MYSQL_CONFIG['user'],
                password=MYSQL_CONFIG['password'],
                database=MYSQL_CONFIG['database'],
                charset=MYSQL_CONFIG.get('charset', 'utf8mb4'),
                autocommit=True,
                cursorclass=self._driver.cursors.DictCursor,
            )
        # mysql-connector-python
        return self._driver.connect(
            host=MYSQL_CONFIG['host'],
            port=MYSQL_CONFIG['port'],
            user=MYSQL_CONFIG['user'],
            password=MYSQL_CONFIG['password'],
            database=MYSQL_CONFIG['database'],
            charset=MYSQL_CONFIG.get('charset', 'utf8mb4'),
            autocommit=True,
        )

    def connect_no_db(self):  # type: ignore[no-untyped-def]
        if self._use_pymysql:
            return self._driver.connect(
                host=MYSQL_CONFIG['host'],
                port=MYSQL_CONFIG['port'],
                user=MYSQL_CONFIG['user'],
                password=MYSQL_CONFIG['password'],
                charset=MYSQL_CONFIG.get('charset', 'utf8mb4'),
                autocommit=True,
                cursorclass=self._driver.cursors.DictCursor,
            )
        return self._driver.connect(
            host=MYSQL_CONFIG['host'],
            port=MYSQL_CONFIG['port'],
            user=MYSQL_CONFIG['user'],
            password=MYSQL_CONFIG['password'],
            charset=MYSQL_CONFIG.get('charset', 'utf8mb4'),
            autocommit=True,
        )


_backend = _DBBackend()


def init_db() -> None:
    """Create database and tables if not present."""
    try:
        conn = _backend.connect()
    except Exception:
        # Try create database then reconnect
        conn = _backend.connect_no_db()
        try:
            with conn.cursor() as cur:
                dbname = MYSQL_CONFIG['database']
                cur.execute(f"CREATE DATABASE IF NOT EXISTS `{dbname}` CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;")
        finally:
            conn.close()
        conn = _backend.connect()

    try:
        with conn.cursor() as cur:
            # Create detections table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS detections (
                  id BIGINT PRIMARY KEY AUTO_INCREMENT,
                  filename VARCHAR(255),
                  passed TINYINT(1) NOT NULL,
                  detected_at DATETIME NOT NULL,
                  centroid_json TEXT,
                  radius_json TEXT,
                  bbox_json TEXT,
                  histogram_json MEDIUMTEXT,
                  process_time_ms INT,
                  username VARCHAR(64) NULL,
                  original_image LONGBLOB NULL,
                  processed_image LONGBLOB NULL,
                  overlay_image LONGBLOB NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
            
            # Create users table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                  id INT PRIMARY KEY AUTO_INCREMENT,
                  username VARCHAR(64) UNIQUE NOT NULL,
                  email VARCHAR(128) UNIQUE NOT NULL,
                  password_hash VARCHAR(64) NOT NULL,
                  department VARCHAR(64) NOT NULL,
                  created_at DATETIME NOT NULL,
                  last_login DATETIME NULL,
                  is_active TINYINT(1) DEFAULT 1
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
    finally:
        conn.close()


def _b64_to_bytes(b64_png: Optional[str]) -> Optional[bytes]:
    if not b64_png:
        return None
    try:
        return base64.b64decode(b64_png)
    except Exception:
        return None


def insert_detection(
    *,
    filename: Optional[str],
    passed: bool,
    detected_at: Optional[datetime.datetime],
    centroid: Optional[List[float]],
    radius: Optional[List[float]],
    bbox: Optional[List[int]],
    histogram: Optional[List[int]],
    process_time_seconds: Optional[float],
    username: Optional[str],
    original_image_b64: Optional[str],
    processed_image_b64: Optional[str],
    overlay_image_b64: Optional[str],
) -> int:
    conn = _backend.connect()
    try:
        with conn.cursor() as cur:
            centroid_json = json.dumps(centroid) if centroid is not None else None
            radius_json = json.dumps(radius) if radius is not None else None
            bbox_json = json.dumps(bbox) if bbox is not None else None
            histogram_json = json.dumps(histogram) if histogram is not None else None

            # Only store images for failures per requirement
            store_images = not passed
            original_blob = _b64_to_bytes(original_image_b64) if store_images else None
            processed_blob = _b64_to_bytes(processed_image_b64) if store_images else None
            overlay_blob = _b64_to_bytes(overlay_image_b64) if store_images else None

            detected_at_dt = detected_at or datetime.datetime.now()
            process_ms = int(process_time_seconds * 1000) if process_time_seconds is not None else None

            sql = (
                "INSERT INTO detections (filename, passed, detected_at, centroid_json, radius_json, bbox_json, "
                "histogram_json, process_time_ms, username, original_image, processed_image, overlay_image) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            )
            cur.execute(
                sql,
                (
                    filename,
                    1 if passed else 0,
                    detected_at_dt.strftime('%Y-%m-%d %H:%M:%S'),
                    centroid_json,
                    radius_json,
                    bbox_json,
                    histogram_json,
                    process_ms,
                    username,
                    original_blob,
                    processed_blob,
                    overlay_blob,
                ),
            )
            return int(cur.lastrowid)
    finally:
        conn.close()


def daily_stats(date_ymd: Optional[str] = None) -> Dict[str, int]:
    """Return counts for the specified date (server local)."""
    if not date_ymd:
        date_ymd = datetime.date.today().strftime('%Y-%m-%d')
    conn = _backend.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  COUNT(*) AS total,
                  SUM(CASE WHEN passed=1 THEN 1 ELSE 0 END) AS passed,
                  SUM(CASE WHEN passed=0 THEN 1 ELSE 0 END) AS failed
                FROM detections
                WHERE DATE(detected_at) = %s
                """,
                (date_ymd,),
            )
            row = cur.fetchone() or {"total": 0, "passed": 0, "failed": 0}
            return {
                "total": int(row.get("total") or 0),
                "passed": int(row.get("passed") or 0),
                "failed": int(row.get("failed") or 0),
            }
    finally:
        conn.close()


def list_failures(
    *,
    date_ymd: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    if not date_ymd:
        date_ymd = datetime.date.today().strftime('%Y-%m-%d')
    conn = _backend.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, filename, detected_at, overlay_image
                FROM detections
                WHERE passed = 0 AND DATE(detected_at) = %s
                ORDER BY detected_at DESC
                LIMIT %s OFFSET %s
                """,
                (date_ymd, int(limit), int(offset)),
            )
            rows = cur.fetchall() or []
            results: List[Dict[str, Any]] = []
            for r in rows:
                overlay = r.get("overlay_image")
                overlay_b64 = base64.b64encode(overlay).decode("ascii") if overlay else None
                results.append(
                    {
                        "id": int(r.get("id")),
                        "filename": r.get("filename"),
                        "detected_at": r.get("detected_at").strftime('%Y-%m-%d %H:%M:%S') if r.get("detected_at") else None,
                        "overlay_image": overlay_b64,
                    }
                )
            return results
    finally:
        conn.close()


def get_failure_image(
    detection_id: int,
    kind: str = "overlay",
) -> Optional[str]:
    """Return base64 PNG of requested image kind for a failure record."""
    column = {
        "original": "original_image",
        "processed": "processed_image",
        "overlay": "overlay_image",
    }.get(kind, "overlay_image")
    conn = _backend.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {column} AS img FROM detections WHERE id = %s AND passed = 0",
                (int(detection_id),),
            )
            row = cur.fetchone()
            if not row:
                return None
            blob = row.get("img")
            if not blob:
                return None
            return base64.b64encode(blob).decode("ascii")
    finally:
        conn.close()


# User management functions
def create_user(username: str, email: str, password_hash: str, department: str) -> bool:
    """Create a new user. Returns True if successful, False if username/email already exists."""
    conn = _backend.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, email, password_hash, department, created_at) VALUES (%s, %s, %s, %s, %s)",
                (username, email, password_hash, department, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            )
            return True
    except Exception:
        return False
    finally:
        conn.close()


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Get user by username. Returns None if not found."""
    conn = _backend.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, email, password_hash, department, created_at, last_login, is_active FROM users WHERE username = %s",
                (username,)
            )
            row = cur.fetchone()
            return row
    finally:
        conn.close()


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get user by email. Returns None if not found."""
    conn = _backend.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, email, password_hash, department, created_at, last_login, is_active FROM users WHERE email = %s",
                (email,)
            )
            row = cur.fetchone()
            return row
    finally:
        conn.close()


def update_last_login(username: str) -> None:
    """Update last login time for user."""
    conn = _backend.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET last_login = %s WHERE username = %s",
                (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), username)
            )
    finally:
        conn.close()


def get_all_users() -> List[Dict[str, Any]]:
    """Get all users (for admin purposes)."""
    conn = _backend.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, email, department, created_at, last_login, is_active FROM users ORDER BY created_at DESC"
            )
            rows = cur.fetchall() or []
            return rows
    finally:
        conn.close()


def get_daily_detection_stats(date_str=None):
    """获取指定日期的检测统计信息"""
    if not date_str:
        date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    
    conn = _backend.connect()
    try:
        with conn.cursor() as cur:
            # 获取指定日期的检测总数
            cur.execute(
                "SELECT COUNT(*) as total FROM detections WHERE DATE(detected_at) = %s",
                (date_str,)
            )
            total_result = cur.fetchone()
            total = total_result['total'] if total_result else 0
            
            # 获取通过数量
            cur.execute(
                "SELECT COUNT(*) as passed FROM detections WHERE DATE(detected_at) = %s AND passed = 1",
                (date_str,)
            )
            passed_result = cur.fetchone()
            passed = passed_result['passed'] if passed_result else 0
            
            # 获取失败数量
            failed = total - passed
            
            # 获取平均处理时间
            cur.execute(
                "SELECT AVG(process_time_ms) as avg_time_ms FROM detections WHERE DATE(detected_at) = %s AND process_time_ms IS NOT NULL",
                (date_str,)
            )
            avg_time_result = cur.fetchone()
            avg_time_ms = float(avg_time_result['avg_time_ms']) if avg_time_result and avg_time_result['avg_time_ms'] else 0.0
            avg_time_seconds = round(avg_time_ms / 1000.0, 3)  # 转换为秒并保留3位小数
            
            return {
                'date': date_str,
                'total': total,
                'passed': passed,
                'failed': failed,
                'avg_time': avg_time_seconds
            }
    except Exception as e:
        print(f"获取每日统计失败: {e}")
        return {
            'date': date_str,
            'total': 0,
            'passed': 0,
            'failed': 0,
            'avg_time': 0.0
        }
    finally:
        conn.close()

def get_weekly_stats(days=7):
    """获取最近几天的统计信息"""
    conn = _backend.connect()
    try:
        with conn.cursor() as cur:
            # 获取最近几天的统计数据
            cur.execute(
                """
                SELECT 
                    DATE(detected_at) as date,
                    COUNT(*) as total,
                    SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as passed,
                    SUM(CASE WHEN passed = 0 THEN 1 ELSE 0 END) as failed,
                    AVG(process_time_ms) as avg_time_ms
                FROM detections 
                WHERE detected_at >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
                GROUP BY DATE(detected_at)
                ORDER BY date ASC
                """,
                (days,)
            )
            
            # 获取数据库中的实际数据
            db_results = {}
            for row in cur.fetchall():
                date_str = row['date'].strftime('%Y-%m-%d')
                avg_time_ms = float(row['avg_time_ms'] or 0)
                avg_time_seconds = round(avg_time_ms / 1000.0, 3)  # 转换为秒并保留3位小数
                
                db_results[date_str] = {
                    'date': date_str,
                    'total': row['total'],
                    'passed': row['passed'],
                    'failed': row['failed'],
                    'avg_time': avg_time_seconds
                }
            
            # 生成完整的7天数据，包括没有检测记录的日期
            results = []
            today = datetime.datetime.now().date()
            for i in range(days - 1, -1, -1):  # 从7天前到今天
                date = today - datetime.timedelta(days=i)
                date_str = date.strftime('%Y-%m-%d')
                
                if date_str in db_results:
                    results.append(db_results[date_str])
                else:
                    # 如果没有该日期的数据，添加零值记录
                    results.append({
                        'date': date_str,
                        'total': 0,
                        'passed': 0,
                        'failed': 0,
                        'avg_time': 0.0
                    })
            
            return results
    except Exception as e:
        print(f"获取周统计失败: {e}")
        return []
    finally:
        conn.close()

def list_detections(limit=50, offset=0, date_from=None, date_to=None):
    """获取检测历史记录"""
    conn = _backend.connect()
    try:
        # 构建查询条件
        where_conditions = []
        params = []
        
        if date_from:
            where_conditions.append("DATE(detected_at) >= %s")
            params.append(date_from)
        
        if date_to:
            where_conditions.append("DATE(detected_at) <= %s")
            params.append(date_to)
        
        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)
        
        # 构建完整的SQL查询
        sql = f"""
            SELECT id, filename, passed, detected_at, username, process_time_ms
            FROM detections
            {where_clause}
            ORDER BY detected_at DESC
            LIMIT %s OFFSET %s
        """
        
        params.extend([limit, offset])
        
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall() or []
            results = []
            for row in rows:
                results.append({
                    'id': int(row.get('id')),
                    'filename': row.get('filename'),
                    'passed': bool(row.get('passed')),
                    'detected_at': row.get('detected_at').strftime('%Y-%m-%d %H:%M:%S') if row.get('detected_at') else None,
                    'username': row.get('username'),
                    'process_time': float(row.get('process_time_ms') or 0) / 1000.0
                })
            return results
    finally:
        conn.close()

def get_history_summary(date_from=None, date_to=None):
    """获取历史统计摘要"""
    conn = _backend.connect()
    try:
        # 构建查询条件
        where_conditions = []
        params = []
        
        if date_from:
            where_conditions.append("DATE(detected_at) >= %s")
            params.append(date_from)
        
        if date_to:
            where_conditions.append("DATE(detected_at) <= %s")
            params.append(date_to)
        
        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)
        
        # 获取总体统计
        sql = f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as passed,
                SUM(CASE WHEN passed = 0 THEN 1 ELSE 0 END) as failed,
                AVG(process_time_ms) as avg_time_ms,
                MIN(detected_at) as first_detection,
                MAX(detected_at) as last_detection
            FROM detections
            {where_clause}
        """
        
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            
            if row:
                avg_time_ms = float(row.get('avg_time_ms') or 0)
                avg_time_seconds = round(avg_time_ms / 1000.0, 3)  # 转换为秒并保留3位小数
                
                return {
                    'total': int(row.get('total') or 0),
                    'passed': int(row.get('passed') or 0),
                    'failed': int(row.get('failed') or 0),
                    'avg_time': avg_time_seconds,
                    'first_detection': row.get('first_detection').strftime('%Y-%m-%d %H:%M:%S') if row.get('first_detection') else None,
                    'last_detection': row.get('last_detection').strftime('%Y-%m-%d %H:%M:%S') if row.get('last_detection') else None,
                    'pass_rate': round((int(row.get('passed') or 0) / int(row.get('total') or 1)) * 100, 2)
                }
            else:
                return {
                    'total': 0,
                    'passed': 0,
                    'failed': 0,
                    'avg_time': 0.0,
                    'first_detection': None,
                    'last_detection': None,
                    'pass_rate': 0.0
                }
    finally:
        conn.close()