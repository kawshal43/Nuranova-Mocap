package lk.nuranova.mocapcamera

import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import lk.nuranova.mocapcamera.protocol.CameraCapabilities
import lk.nuranova.mocapcamera.protocol.ClockSyncMath
import lk.nuranova.mocapcamera.protocol.ClockSyncReportPayload
import lk.nuranova.mocapcamera.protocol.PairRequest
import lk.nuranova.mocapcamera.protocol.PairingUriParser
import lk.nuranova.mocapcamera.protocol.RecordingConfig
import lk.nuranova.mocapcamera.protocol.StudioBeaconMessage
import lk.nuranova.mocapcamera.protocol.SupportedMode
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class ClockSyncAndProtocolTest {

    @Test
    fun clockSyncRejectsHighRttOutliers() {
        val trueOffsetNs = 5_000_000_000L
        val samples = (0 until 16).map { seq ->
            val t1 = 10_000_000_000L + seq * 100_000_000L
            val rtt = if (seq == 3 || seq == 11) 60_000_000L else 1_600_000L
            val mid = t1 + rtt / 2L
            val t2 = mid + trueOffsetNs - 50_000L
            val t3 = t2 + 100_000L
            val t4 = t1 + rtt
            ClockSyncReportPayload(seq, t1, t2, t3, t4)
        }
        val summary = ClockSyncMath.summarize(samples)
        assertTrue("Expected residual < 2 ms, got ${summary.residualMs}", summary.residualMs < 2.0)
        assertTrue("Expected offset error < 1 ms", abs(summary.offsetNs - trueOffsetNs) < 1_000_000.0)
    }

    @Test
    fun protocolSerializationMatchesDesktopSchema() {
        val json = Json { encodeDefaults = true; ignoreUnknownKeys = true }
        val req = PairRequest(
            deviceId = "android_test_01",
            deviceName = "Test Phone",
            pin = "123456",
            capabilities = CameraCapabilities(
                manufacturer = "Generic",
                model = "ModelX",
                supportedModes = listOf(SupportedMode(1920, 1080, listOf(listOf(60, 60)), 60.0))
            )
        )
        val encoded = json.encodeToString(req)
        assertTrue(encoded.contains("\"protocol_version\":\"1.0.0\""))
        assertTrue(encoded.contains("\"device_id\":\"android_test_01\""))
        val decoded = json.decodeFromString(PairRequest.serializer(), encoded)
        assertEquals(req.deviceId, decoded.deviceId)

        val cfg = RecordingConfig()
        assertEquals(1920, cfg.width)
        assertEquals(1080, cfg.height)
        assertEquals(60.0, cfg.targetFps, 1e-6)
    }

    @Test
    fun qrPairingUriAndUdpBeaconParseCleanly() {
        val uri = "nuranova://pair?host=192.168.8.124&port=8765&pin=482910&name=Nuranova+Mocap+Studio"
        val parsed = PairingUriParser.parse(uri)
        assertNotNull(parsed)
        assertEquals("192.168.8.124", parsed!!.host)
        assertEquals(8765, parsed.port)
        assertEquals("482910", parsed.pin)
        assertEquals("Nuranova Mocap Studio", parsed.studioName)
        assertNull(PairingUriParser.parse("https://example.com"))

        val json = Json { ignoreUnknownKeys = true }
        val beaconRaw = """{"type":"nuranova_studio_beacon","protocol_version":"1.0.0","studio_name":"Nuranova Mocap Studio","host":"192.168.8.124","all_hosts":["192.168.8.124"],"port":8765}"""
        val beacon = json.decodeFromString(StudioBeaconMessage.serializer(), beaconRaw)
        assertEquals("nuranova_studio_beacon", beacon.type)
        assertEquals("192.168.8.124", beacon.host)
        assertEquals(8765, beacon.port)
    }
}
