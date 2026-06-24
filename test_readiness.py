import sys
import os
# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import unittest
from unittest.mock import MagicMock, patch
import asyncio
from datetime import datetime, timedelta

# Import our application components
from config.settings import settings
from services.intent_service import IntentService, IntentResult
from services.cache_service import CacheService
from services.sql_service import SQLService, PROCEDURE_REGISTRY
from services.manufacturing_service import ManufacturingService
from app import app
from fastapi.testclient import TestClient


class TestSettings(unittest.TestCase):
    def test_sql_connection_string(self):
        self.assertIn("DRIVER=", settings.sql_connection_string)
        self.assertIn("SERVER=", settings.sql_connection_string)
        self.assertIn("DATABASE=", settings.sql_connection_string)


class TestIntentService(unittest.TestCase):
    def setUp(self):
        self.intent_service = IntentService()
        self.orig_registry = PROCEDURE_REGISTRY.copy()
        PROCEDURE_REGISTRY.clear()
        PROCEDURE_REGISTRY.update({
            "SP_MPAS_CREATED_ORDERS": {
                "description": "Returns status and stage progress of active orders",
                "category": "order",
                "intent": "order_status",
                "params": ["@OrderNo", "@SerialNo"]
            },
            "SP_MPAS_HOURLY_PRODUCTION_METRICS": {
                "description": "Returns hourly production metrics per line",
                "category": "production",
                "intent": "production_metrics",
                "params": ["@LineCode", "@ShiftDate", "@ShiftID"]
            },
            "SP_MPAS_PROCESS_STATUS_TORQUE": {
                "description": "Returns torque data TorqueBolt1-38",
                "category": "quality",
                "intent": "torque_quality",
                "params": ["@OrderNo", "@SerialNo"]
            },
            "SP_MPAS_BYPASS_AUDIT": {
                "description": "Returns bypass request and approval audit trail",
                "category": "bypass",
                "intent": "bypass",
                "params": ["@OrderNo"]
            },
            "PNR_Get_updateProcessData_API_Raw_Data": {
                "description": "Returns process execution data",
                "category": "process",
                "intent": "process_results",
                "params": ["@OrderNo"]
            }
        })
        import services.sql_service
        services.sql_service.APPROVED_PROCEDURES = set(PROCEDURE_REGISTRY.keys())

    def tearDown(self):
        PROCEDURE_REGISTRY.clear()
        PROCEDURE_REGISTRY.update(self.orig_registry)
        import services.sql_service
        services.sql_service.APPROVED_PROCEDURES = set(PROCEDURE_REGISTRY.keys())

    def test_entity_extraction_order_no(self):
        result = self.intent_service.classify("status of order 147190737")
        self.assertEqual(result.entities["order_no"], "147190737")

    def test_entity_extraction_serial_axle(self):
        result = self.intent_service.classify("axle RHPW133610")
        self.assertEqual(result.entities["serial_no"], "RHPW133610")

    def test_entity_extraction_serial_truck(self):
        result = self.intent_service.classify("truck TRPS706859")
        self.assertEqual(result.entities["serial_no"], "TRPS706859")

    def test_entity_extraction_serial_vin(self):
        result = self.intent_service.classify("vin MB1G9VHD1TPRK3407")
        self.assertEqual(result.entities["serial_no"], "MB1G9VHD1TPRK3407")

    def test_entity_extraction_line_code(self):
        result1 = self.intent_service.classify("line 3 status")
        result2 = self.intent_service.classify("L1 production metrics")
        self.assertEqual(result1.entities["line_code"], "3")
        self.assertEqual(result2.entities["line_code"], "1")

    def test_entity_extraction_shift_and_date(self):
        result = self.intent_service.classify("shift A on 2026-06-21")
        self.assertEqual(result.entities["shift_id"], "A")
        self.assertEqual(result.entities["shift_date"], "2026-06-21")

    def test_intent_classification_order_status(self):
        result = self.intent_service.classify("what is the status of order 147190737")
        self.assertEqual(result.intent, "order_status")
        self.assertEqual(result.procedure, "SP_MPAS_CREATED_ORDERS")

    def test_intent_classification_production_metrics(self):
        result = self.intent_service.classify("how many units did line 3 produce this shift")
        self.assertEqual(result.intent, "production_metrics")
        self.assertEqual(result.procedure, "SP_MPAS_HOURLY_PRODUCTION_METRICS")

    def test_intent_classification_torque_quality(self):
        result = self.intent_service.classify("torque on bolt 5 for RHPW133610")
        self.assertEqual(result.intent, "torque_quality")
        self.assertEqual(result.procedure, "SP_MPAS_PROCESS_STATUS_TORQUE")

    def test_intent_classification_bypass(self):
        result = self.intent_service.classify("bypass history for order 147190737")
        self.assertEqual(result.intent, "bypass")
        self.assertEqual(result.procedure, "SP_MPAS_BYPASS_AUDIT")

    def test_intent_classification_live_opc(self):
        result = self.intent_service.classify("what is the current machine status right now")
        self.assertEqual(result.intent, "live_opc")
        self.assertIsNone(result.procedure)
        self.assertTrue(result.use_opc)

    def test_intent_classification_document_search(self):
        result = self.intent_service.classify("show me the sop or instruction manual")
        self.assertEqual(result.intent, "document_search")
        self.assertIsNone(result.procedure)
        self.assertTrue(result.use_rag)

    def test_clarification_needed(self):
        result = self.intent_service.classify("show process results")
        self.assertEqual(result.intent, "process_results")
        self.assertTrue(result.clarification_needed)
        self.assertIn("order number", result.clarification_prompt)


class TestCacheService(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cache = CacheService()

    async def test_update_and_get(self):
        await self.cache.update("Line1.MachineStatus", "RUNNING")
        val = await self.cache.get("Line1.MachineStatus")
        self.assertEqual(val, "RUNNING")

    async def test_bulk_update_and_snapshot(self):
        tags = {
            "Line1.MachineStatus": "IDLE",
            "Line1.ShiftActualCount": 100
        }
        await self.cache.bulk_update(tags)
        snapshot = await self.cache.snapshot()
        
        self.assertEqual(snapshot["Line1.MachineStatus"]["value"], "IDLE")
        self.assertEqual(snapshot["Line1.ShiftActualCount"]["value"], 100)
        self.assertEqual(self.cache.tag_count, 2)
        self.assertIsNotNone(self.cache.last_update)

    async def test_staleness_check(self):
        await self.cache.update("Line1.MachineStatus", "FAULT")
        # Newly updated should not be stale
        self.assertFalse(self.cache.is_stale("Line1.MachineStatus", max_age_sec=5.0))
        
        # Override timestamp to mock stale state
        async with self.cache._lock:
            self.cache._timestamps["Line1.MachineStatus"] = datetime.now() - timedelta(seconds=10)
        
        self.assertTrue(self.cache.is_stale("Line1.MachineStatus", max_age_sec=5.0))


class TestSQLService(unittest.TestCase):
    def setUp(self):
        self.sql_service = SQLService()
        self.orig_registry = PROCEDURE_REGISTRY.copy()
        PROCEDURE_REGISTRY.clear()
        PROCEDURE_REGISTRY.update({
            "SP_MPAS_CREATED_ORDERS": {
                "description": "Returns status and stage progress of active orders",
                "category": "order",
                "intent": "order_status",
                "params": ["@OrderNo", "@SerialNo"]
            }
        })
        import services.sql_service
        services.sql_service.APPROVED_PROCEDURES = set(PROCEDURE_REGISTRY.keys())

    def tearDown(self):
        PROCEDURE_REGISTRY.clear()
        PROCEDURE_REGISTRY.update(self.orig_registry)
        import services.sql_service
        services.sql_service.APPROVED_PROCEDURES = set(PROCEDURE_REGISTRY.keys())

    def test_unapproved_procedure_fails(self):
        with self.assertRaises(PermissionError):
            self.sql_service._validate_proc("SP_UNAPPROVED_DELETE_ALL")

    def test_approved_procedure_passes(self):
        # Should not raise exception
        self.sql_service._validate_proc("SP_MPAS_CREATED_ORDERS")

    def test_registry_summary(self):
        summary = self.sql_service.registry_summary()
        self.assertEqual(len(summary), len(PROCEDURE_REGISTRY))
        procedures_in_summary = [s["procedure"] for s in summary]
        self.assertIn("SP_MPAS_CREATED_ORDERS", procedures_in_summary)


class TestApiRoutes(unittest.TestCase):
    def setUp(self):
        self.orig_registry = PROCEDURE_REGISTRY.copy()
        PROCEDURE_REGISTRY.clear()
        PROCEDURE_REGISTRY.update({
            "SP_MPAS_CREATED_ORDERS": {
                "description": "Returns status and stage progress of active orders",
                "category": "order",
                "intent": "order_status",
                "params": ["@OrderNo", "@SerialNo"]
            }
        })
        import services.sql_service
        services.sql_service.APPROVED_PROCEDURES = set(PROCEDURE_REGISTRY.keys())

    def tearDown(self):
        PROCEDURE_REGISTRY.clear()
        PROCEDURE_REGISTRY.update(self.orig_registry)
        import services.sql_service
        services.sql_service.APPROVED_PROCEDURES = set(PROCEDURE_REGISTRY.keys())

    def test_root_endpoint(self):
        with TestClient(app) as client:
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/html", response.headers["content-type"])

    def test_setting_endpoint(self):
        with TestClient(app) as client:
            response = client.get("/setting")
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/html", response.headers["content-type"])

    def test_health_endpoint(self):
        with TestClient(app) as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "ok")
            self.assertIn("services", data)

    def test_get_config_endpoint(self):
        with TestClient(app) as client:
            response = client.get("/api/config")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["plant_name"], settings.PLANT_NAME)
            self.assertEqual(data["llm_model"], settings.OLLAMA_MODEL)

    def test_list_procedures_endpoint(self):
        with TestClient(app) as client:
            response = client.get("/api/admin/procedures")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("procedures", data)
            self.assertTrue(len(data["procedures"]) > 0)

    @patch("pyodbc.connect")
    def test_sql_databases_list_endpoint(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [("Production_DB",), ("Test_DB",)]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        with TestClient(app) as client:
            response = client.post(
                "/api/admin/sql/databases",
                json={
                    "server": "localhost",
                    "username": "dbuser",
                    "password": "password",
                    "driver": "ODBC Driver 17 for SQL Server"
                }
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["databases"], ["Production_DB", "Test_DB"])

    @patch("pyodbc.connect")
    def test_sql_databases_list_endpoint_error(self, mock_connect):
        mock_connect.side_effect = Exception("Connection refused")

        with TestClient(app) as client:
            response = client.post(
                "/api/admin/sql/databases",
                json={
                    "server": "localhost",
                    "username": "dbuser",
                    "password": "password",
                    "driver": "ODBC Driver 17 for SQL Server"
                }
            )
            self.assertEqual(response.status_code, 400)
            data = response.json()
            self.assertEqual(data["status"], "error")
            self.assertIn("Connection refused", data["error"])


class TestVectorServiceAndNewEndpoints(unittest.TestCase):
    def setUp(self):
        self.orig_sql_server = settings.SQL_SERVER
        self.orig_sql_db = settings.SQL_DATABASE
        self.orig_sql_username = settings.SQL_USERNAME
        self.orig_sql_password = settings.SQL_PASSWORD
        self.orig_sql_driver = settings.SQL_DRIVER
        self.orig_sql_timeout = settings.SQL_TIMEOUT
        self.orig_kb_topk = settings.CHROMA_TOP_K
        self.orig_kb_chunk_size = settings.KB_CHUNK_SIZE
        self.orig_kb_overlap = settings.KB_CHUNK_OVERLAP
        self.orig_kb_upload_dir = settings.KB_UPLOAD_DIR

    def tearDown(self):
        settings.SQL_SERVER = self.orig_sql_server
        settings.SQL_DATABASE = self.orig_sql_db
        settings.SQL_USERNAME = self.orig_sql_username
        settings.SQL_PASSWORD = self.orig_sql_password
        settings.SQL_DRIVER = self.orig_sql_driver
        settings.SQL_TIMEOUT = self.orig_sql_timeout
        settings.CHROMA_TOP_K = self.orig_kb_topk
        settings.KB_CHUNK_SIZE = self.orig_kb_chunk_size
        settings.KB_CHUNK_OVERLAP = self.orig_kb_overlap
        settings.KB_UPLOAD_DIR = self.orig_kb_upload_dir

        from config.settings import update_env_values
        update_env_values({
            "SQL_SERVER": self.orig_sql_server,
            "SQL_DATABASE": self.orig_sql_db,
            "SQL_USERNAME": self.orig_sql_username,
            "SQL_PASSWORD": self.orig_sql_password,
            "SQL_DRIVER": self.orig_sql_driver,
            "SQL_TIMEOUT": str(self.orig_sql_timeout),
            "CHROMA_TOP_K": str(self.orig_kb_topk),
            "KB_CHUNK_SIZE": str(self.orig_kb_chunk_size),
            "KB_CHUNK_OVERLAP": str(self.orig_kb_overlap),
            "KB_UPLOAD_DIR": self.orig_kb_upload_dir,
        })

    def test_sql_config_save_empty_strings(self):
        with TestClient(app) as client:
            response = client.post(
                "/api/admin/sql/config",
                json={
                    "server": "",
                    "database": "",
                    "username": "",
                    "password": "",
                    "driver": "ODBC Driver 17 for SQL Server",
                    "timeout": 30
                }
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(settings.SQL_SERVER, "")
            self.assertEqual(settings.SQL_DATABASE, "")

    def test_kb_config_custom_values(self):
        with TestClient(app) as client:
            response = client.post(
                "/api/admin/kb/config",
                json={
                    "top_k": 8,
                    "chunk_size": 500,
                    "chunk_overlap": 50,
                    "upload_dir": "D:\\Siuuu Manual"
                }
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(settings.CHROMA_TOP_K, 8)
            self.assertEqual(settings.KB_CHUNK_SIZE, 500)
            self.assertEqual(settings.KB_CHUNK_OVERLAP, 50)
            self.assertEqual(settings.KB_UPLOAD_DIR, "D:\\Siuuu Manual")


if __name__ == "__main__":
    unittest.main()
