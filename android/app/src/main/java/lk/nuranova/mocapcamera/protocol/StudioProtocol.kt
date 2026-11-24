package lk.nuranova.mocapcamera.protocol

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import java.net.URLDecoder
import kotlin.math.sqrt

const val PROTOCOL_VERSION = "1.0.0"
const val MDNS_SERVICE_TYPE = "_nuranova-mocap._tcp."
const val DEFAULT_STUDIO_PORT = 8765
const val UDP_DISCOVERY_PORT = 8766

@Serializable
data class StudioBeaconMessage(
    val type: String = "",
    @SerialName("protocol_version") val protocolVersion: String = PROTOCOL_VERSION,
    @SerialName("studio_name") val studioName: String = "Nuranova Mocap Studio",
    val host: String = "",
    @SerialName("all_hosts") val allHosts: List<String> = emptyList(),
    val port: Int = DEFAULT_STUDIO_PORT
)

data class PairingQrPayload(
    val host: String,
    val port: Int,
    val pin: String,
    val studioName: String = "Nuranova Mocap Studio"
)

object PairingUriParser {
    /**
     * Parse QR code URIs like:
     * `nuranova://pair?host=192.168.8.124&port=8765&pin=482910&name=Nuranova+Mocap+Studio`
     */
    fun parse(raw: String): PairingQrPayload? {
        val text = raw.trim()
        if (!text.startsWith("nuranova://pair", ignoreCase = true)) return null
        val qIdx = text.indexOf('?')
        if (qIdx < 0 || qIdx >= text.length - 1) return null
        val query = text.substring(qIdx + 1)
        val params = mutableMapOf<String, String>()
        for (pair in query.split('&')) {
            val eq = pair.indexOf('=')
            if (eq > 0) {
                val k = URLDecoder.decode(pair.substring(0, eq), "UTF-8").lowercase()
                val v = URLDecoder.decode(pair.substring(eq + 1), "UTF-8")
                params[k] = v
            }
        }
        val host = params["host"]?.trim().orEmpty()
        val pin = params["pin"]?.trim().orEmpty()
        val port = params["port"]?.toIntOrNull() ?: DEFAULT_STUDIO_PORT
        val name = params["name"]?.trim().orEmpty().ifBlank { "Nuranova Mocap Studio" }
        if (host.isBlank() || pin.length < 6) return null
        return PairingQrPayload(host = host, port = port, pin = pin, studioName = name)
    }
}

@Serializable
data class SupportedMode(
    val width: Int,
    val height: Int,
    @SerialName("fps_ranges") val fpsRanges: List<List<Int>> = emptyList(),
    @SerialName("max_fps") val maxFps: Double,
    @SerialName("supports_high_speed") val supportsHighSpeed: Boolean = false
)

@Serializable
data class CameraCapabilities(
    val manufacturer: String = "Unknown",
    val model: String = "Unknown",
    @SerialName("android_api_level") val androidApiLevel: Int = 26,
    @SerialName("camera_id") val cameraId: String = "0",
    @SerialName("lens_facing") val lensFacing: String = "back",
    @SerialName("hardware_level") val hardwareLevel: String = "LIMITED",
    @SerialName("timestamp_source") val timestampSource: String = "UNKNOWN",
    @SerialName("supported_modes") val supportedModes: List<SupportedMode> = emptyList(),
    @SerialName("supports_manual_exposure") val supportsManualExposure: Boolean = false,
    @SerialName("supports_manual_focus") val supportsManualFocus: Boolean = false,
    @SerialName("supports_awb_lock") val supportsAwbLock: Boolean = false,
    @SerialName("supports_video_stabilization_off") val supportsVideoStabilizationOff: Boolean = true,
    @SerialName("exposure_time_range_ns") val exposureTimeRangeNs: List<Long>? = null,
    @SerialName("iso_range") val isoRange: List<Int>? = null,
    @SerialName("focal_lengths_mm") val focalLengthsMm: List<Float> = emptyList(),
    @SerialName("sensor_physical_size_mm") val sensorPhysicalSizeMm: List<Float>? = null
)

@Serializable
data class RecordingConfig(
    val width: Int = 1920,
    val height: Int = 1080,
    @SerialName("target_fps") val targetFps: Double = 60.0,
    @SerialName("bitrate_bps") val bitrateBps: Int = 20_000_000,
    val codec: String = "H264",
    @SerialName("camera_lens") val cameraLens: String = "rear_main",
    @SerialName("lock_focus") val lockFocus: Boolean = true,
    @SerialName("focus_distance_diopters") val focusDistanceDiopters: Float? = null,
    @SerialName("manual_exposure") val manualExposure: Boolean = true,
    @SerialName("shutter_speed_ns") val shutterSpeedNs: Long = 4_000_000L,
    val iso: Int? = null,
    @SerialName("lock_white_balance") val lockWhiteBalance: Boolean = true,
    @SerialName("disable_stabilization") val disableStabilization: Boolean = true,
    @SerialName("disable_hdr") val disableHdr: Boolean = true,
    @SerialName("record_audio") val recordAudio: Boolean = true,
    @SerialName("preview_width") val previewWidth: Int = 320,
    @SerialName("preview_height") val previewHeight: Int = 180,
    @SerialName("preview_fps") val previewFps: Double = 8.0,
    @SerialName("preview_jpeg_quality") val previewJpegQuality: Int = 55
)

@Serializable
data class DeviceTelemetry(
    @SerialName("device_id") val deviceId: String,
    @SerialName("battery_percent") val batteryPercent: Double,
    @SerialName("is_charging") val isCharging: Boolean = false,
    @SerialName("temperature_celsius") val temperatureCelsius: Double? = null,
    @SerialName("available_storage_mb") val availableStorageMb: Double,
    @SerialName("actual_width") val actualWidth: Int = 1920,
    @SerialName("actual_height") val actualHeight: Int = 1080,
    @SerialName("measured_fps") val measuredFps: Double = 0.0,
    @SerialName("dropped_frames") val droppedFrames: Int = 0,
    @SerialName("shutter_speed_ns") val shutterSpeedNs: Long? = null,
    val iso: Int? = null,
    @SerialName("focus_locked") val focusLocked: Boolean = false,
    @SerialName("wb_locked") val wbLocked: Boolean = false,
    @SerialName("orientation_degrees") val orientationDegrees: Int = 0,
    @SerialName("recording_state") val recordingState: String = "idle",
    @SerialName("active_take_id") val activeTakeId: String? = null,
    @SerialName("recording_duration_s") val recordingDurationS: Double = 0.0,
    @SerialName("monotonic_time_ns") val monotonicTimeNs: Long = 0L
)

@Serializable
data class PairRequest(
    @SerialName("protocol_version") val protocolVersion: String = PROTOCOL_VERSION,
    @SerialName("device_id") val deviceId: String,
    @SerialName("device_name") val deviceName: String,
    val pin: String,
    val capabilities: CameraCapabilities,
    val role: String = "body"
)

@Serializable
data class PairResponse(
    @SerialName("protocol_version") val protocolVersion: String = PROTOCOL_VERSION,
    val paired: Boolean,
    @SerialName("device_id") val deviceId: String,
    @SerialName("assigned_camera_uid") val assignedCameraUid: String,
    @SerialName("auth_token") val authToken: String,
    @SerialName("studio_name") val studioName: String = "Nuranova Mocap Studio",
    @SerialName("studio_time_ns") val studioTimeNs: Long = 0L
)

@Serializable
data class ClockSyncRequest(
    @SerialName("device_id") val deviceId: String,
    val seq: Int,
    @SerialName("client_t1_ns") val clientT1Ns: Long
)

@Serializable
data class ClockSyncResponse(
    @SerialName("device_id") val deviceId: String,
    val seq: Int,
    @SerialName("client_t1_ns") val clientT1Ns: Long,
    @SerialName("server_t2_ns") val serverT2Ns: Long,
    @SerialName("server_t3_ns") val serverT3Ns: Long
)

@Serializable
data class ClockSyncReportPayload(
    val seq: Int,
    @SerialName("t1_client_ns") val t1ClientNs: Long,
    @SerialName("t2_server_ns") val t2ServerNs: Long,
    @SerialName("t3_server_ns") val t3ServerNs: Long,
    @SerialName("t4_client_ns") val t4ClientNs: Long
)

@Serializable
data class ScheduleRecordCommand(
    val command: String = "SCHEDULE_RECORD",
    @SerialName("session_id") val sessionId: String,
    @SerialName("take_id") val takeId: String,
    @SerialName("scheduled_start_studio_ns") val scheduledStartStudioNs: Long,
    @SerialName("scheduled_start_device_ns") val scheduledStartDeviceNs: Long,
    val config: RecordingConfig = RecordingConfig(),
    @SerialName("flash_sync_pulse") val flashSyncPulse: Boolean = false,
    @SerialName("audio_clap_cue") val audioClapCue: Boolean = true
)

@Serializable
data class StopRecordCommand(
    val command: String = "STOP_RECORD",
    @SerialName("session_id") val sessionId: String,
    @SerialName("take_id") val takeId: String
)

@Serializable
data class FrameTimestampRecord(
    @SerialName("frame_index") val frameIndex: Int,
    @SerialName("sensor_timestamp_ns") val sensorTimestampNs: Long,
    @SerialName("pts_us") val ptsUs: Long,
    @SerialName("monotonic_receive_ns") val monotonicReceiveNs: Long,
    @SerialName("exposure_ns") val exposureNs: Long? = null,
    @SerialName("rolling_shutter_skew_ns") val rollingShutterSkewNs: Long? = null,
    @SerialName("frame_duration_ns") val frameDurationNs: Long? = null
)

@Serializable
data class TransferInitRequest(
    @SerialName("device_id") val deviceId: String,
    @SerialName("session_id") val sessionId: String,
    @SerialName("take_id") val takeId: String,
    val filename: String,
    @SerialName("file_kind") val fileKind: String,
    @SerialName("total_size_bytes") val totalSizeBytes: Long,
    @SerialName("sha256_hex") val sha256Hex: String
)

@Serializable
data class TransferInitResponse(
    @SerialName("transfer_id") val transferId: String,
    @SerialName("received_bytes") val receivedBytes: Long,
    @SerialName("chunk_size_bytes") val chunkSizeBytes: Int = 1024 * 1024,
    @SerialName("already_complete") val alreadyComplete: Boolean = false
)

@Serializable
data class TransferStatusResponse(
    @SerialName("transfer_id") val transferId: String,
    @SerialName("received_bytes") val receivedBytes: Long,
    @SerialName("total_size_bytes") val totalSizeBytes: Long,
    val verified: Boolean,
    @SerialName("computed_sha256") val computedSha256: String? = null,
    @SerialName("expected_sha256") val expectedSha256: String,
    @SerialName("can_delete_local_copy") val canDeleteLocalCopy: Boolean = false
)

data class ClockSyncSummary(
    val offsetNs: Double,
    val minRttMs: Double,
    val residualMs: Double,
    val sampleCount: Int
)

/**
 * Computes robust lowest-RTT NTP-style clock offset (studio_ns - device_ns) on Android.
 */
object ClockSyncMath {
    fun summarize(samples: List<ClockSyncReportPayload>, keepFraction: Double = 0.6): ClockSyncSummary {
        require(samples.isNotEmpty()) { "At least one sync sample is required." }
        val sorted = samples.sortedBy {
            ((it.t4ClientNs - it.t1ClientNs) - (it.t3ServerNs - it.t2ServerNs)).coerceAtLeast(0L)
        }
        val keepN = (sorted.size * keepFraction).toInt().coerceIn(1, sorted.size)
        val kept = sorted.take(keepN)
        val offsets = kept.map { s ->
            0.5 * (s.t2ServerNs + s.t3ServerNs) - 0.5 * (s.t1ClientNs + s.t4ClientNs)
        }.sorted()
        val medianOffset = offsets[offsets.size / 2]
        val minRttMs = ((sorted.first().t4ClientNs - sorted.first().t1ClientNs) -
            (sorted.first().t3ServerNs - sorted.first().t2ServerNs)).coerceAtLeast(0L) * 1e-6
        val meanSqMs = offsets.map { ((it - medianOffset) * 1e-6).let { d -> d * d } }.average()
        val residualMs = sqrt(meanSqMs + (0.25 * minRttMs) * (0.25 * minRttMs))
        return ClockSyncSummary(
            offsetNs = medianOffset,
            minRttMs = minRttMs,
            residualMs = residualMs,
            sampleCount = samples.size
        )
    }
}
