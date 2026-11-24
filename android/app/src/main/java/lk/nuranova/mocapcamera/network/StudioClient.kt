package lk.nuranova.mocapcamera.network

import android.content.Context
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import android.os.SystemClock
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import lk.nuranova.mocapcamera.protocol.ClockSyncMath
import lk.nuranova.mocapcamera.protocol.ClockSyncReportPayload
import lk.nuranova.mocapcamera.protocol.ClockSyncRequest
import lk.nuranova.mocapcamera.protocol.ClockSyncResponse
import lk.nuranova.mocapcamera.protocol.ClockSyncSummary
import lk.nuranova.mocapcamera.protocol.DeviceTelemetry
import lk.nuranova.mocapcamera.protocol.MDNS_SERVICE_TYPE
import lk.nuranova.mocapcamera.protocol.PairRequest
import lk.nuranova.mocapcamera.protocol.PairResponse
import lk.nuranova.mocapcamera.protocol.ScheduleRecordCommand
import lk.nuranova.mocapcamera.protocol.StopRecordCommand
import lk.nuranova.mocapcamera.protocol.TransferInitRequest
import lk.nuranova.mocapcamera.protocol.TransferInitResponse
import lk.nuranova.mocapcamera.protocol.TransferStatusResponse
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString.Companion.toByteString
import java.io.File
import java.io.RandomAccessFile
import java.util.concurrent.TimeUnit

data class DiscoveredStudio(
    val serviceName: String,
    val host: String,
    val port: Int
)

/**
 * Local-LAN client handling Android NSD discovery, PIN pairing, authenticated WebSocket control,
 * repeated NTP clock synchronization, telemetry heartbeats, and resumable SHA-256 uploads.
 */
class StudioClient(private val context: Context) {

    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }
    private val jsonMedia = "application/json; charset=utf-8".toMediaType()
    private val octetMedia = "application/octet-stream".toMediaType()

    private val http = OkHttpClient.Builder()
        .connectTimeout(6, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .pingInterval(15, TimeUnit.SECONDS)
        .build()

    private var webSocket: WebSocket? = null
    private var nsdManager: NsdManager? = null
    private var discoveryListener: NsdManager.DiscoveryListener? = null

    var onScheduleRecord: ((ScheduleRecordCommand) -> Unit)? = null
    var onStopRecord: ((StopRecordCommand) -> Unit)? = null
    var onConnectionStateChanged: ((Boolean, String) -> Unit)? = null

    @Suppress("DEPRECATION")
    fun startDiscovery(onStudioFound: (DiscoveredStudio) -> Unit) {
        stopDiscovery()
        val mgr = context.getSystemService(Context.NSD_SERVICE) as? NsdManager ?: return
        nsdManager = mgr
        val listener = object : NsdManager.DiscoveryListener {
            override fun onDiscoveryStarted(regType: String) {}
            override fun onServiceFound(serviceInfo: NsdServiceInfo) {
                if (serviceInfo.serviceType.contains("nuranova-mocap")) {
                    mgr.resolveService(serviceInfo, object : NsdManager.ResolveListener {
                        override fun onResolveFailed(info: NsdServiceInfo, errorCode: Int) {}
                        override fun onServiceResolved(info: NsdServiceInfo) {
                            val ip = info.host?.hostAddress ?: return
                            onStudioFound(DiscoveredStudio(info.serviceName, ip, info.port))
                        }
                    })
                }
            }
            override fun onServiceLost(serviceInfo: NsdServiceInfo) {}
            override fun onDiscoveryStopped(serviceType: String) {}
            override fun onStartDiscoveryFailed(serviceType: String, errorCode: Int) {}
            override fun onStopDiscoveryFailed(serviceType: String, errorCode: Int) {}
        }
        discoveryListener = listener
        runCatching { mgr.discoverServices(MDNS_SERVICE_TYPE, NsdManager.PROTOCOL_DNS_SD, listener) }
    }

    fun stopDiscovery() {
        val mgr = nsdManager ?: return
        val listener = discoveryListener ?: return
        runCatching { mgr.stopServiceDiscovery(listener) }
        discoveryListener = null
    }

    suspend fun checkHealth(host: String, port: Int): Result<String> = withContext(Dispatchers.IO) {
        runCatching {
            val req = Request.Builder().url("http://$host:$port/api/v1/health").get().build()
            http.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) error("HTTP ${resp.code}")
                resp.body?.string() ?: "{}"
            }
        }
    }

    suspend fun pair(host: String, port: Int, pairReq: PairRequest): Result<PairResponse> =
        withContext(Dispatchers.IO) {
            runCatching {
                val body = json.encodeToString(pairReq).toRequestBody(jsonMedia)
                val req = Request.Builder().url("http://$host:$port/api/v1/pair").post(body).build()
                http.newCall(req).execute().use { resp ->
                    val text = resp.body?.string().orEmpty()
                    if (!resp.isSuccessful) error("Pairing rejected (HTTP ${resp.code}): $text")
                    json.decodeFromString(PairResponse.serializer(), text)
                }
            }
        }

    fun connectWebSocket(host: String, port: Int, deviceId: String, authToken: String) {
        disconnectWebSocket()
        val req = Request.Builder()
            .url("ws://$host:$port/ws/v1/camera/$deviceId?token=$authToken")
            .header("Authorization", "Bearer $authToken")
            .build()

        webSocket = http.newWebSocket(req, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                onConnectionStateChanged?.invoke(true, "Connected to $host:$port")
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                runCatching {
                    val obj = json.parseToJsonElement(text).jsonObject
                    when (obj["command"]?.jsonPrimitive?.content) {
                        "SCHEDULE_RECORD" -> {
                            val cmd = json.decodeFromString(ScheduleRecordCommand.serializer(), text)
                            onScheduleRecord?.invoke(cmd)
                        }
                        "STOP_RECORD" -> {
                            val cmd = json.decodeFromString(StopRecordCommand.serializer(), text)
                            onStopRecord?.invoke(cmd)
                        }
                    }
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                onConnectionStateChanged?.invoke(false, "Disconnected: ${t.message}")
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                onConnectionStateChanged?.invoke(false, "Closed ($code)")
            }
        })
    }

    fun disconnectWebSocket() {
        runCatching { webSocket?.close(1000, "Client disconnect") }
        webSocket = null
    }

    fun sendPreviewFrame(jpegBytes: ByteArray) {
        webSocket?.send(jpegBytes.toByteString())
    }

    suspend fun postTelemetry(host: String, port: Int, token: String, telemetry: DeviceTelemetry) =
        withContext(Dispatchers.IO) {
            runCatching {
                val body = json.encodeToString(telemetry).toRequestBody(jsonMedia)
                val req = Request.Builder()
                    .url("http://$host:$port/api/v1/telemetry")
                    .header("Authorization", "Bearer $token")
                    .header("X-Device-Id", telemetry.deviceId)
                    .post(body)
                    .build()
                http.newCall(req).execute().close()
            }
        }

    suspend fun runClockSyncBurst(
        host: String,
        port: Int,
        deviceId: String,
        token: String,
        probes: Int = 16
    ): Result<ClockSyncSummary> = withContext(Dispatchers.IO) {
        runCatching {
            val samples = mutableListOf<ClockSyncReportPayload>()
            for (seq in 0 until probes) {
                val t1 = SystemClock.elapsedRealtimeNanos()
                val reqPayload = ClockSyncRequest(deviceId = deviceId, seq = seq, clientT1Ns = t1)
                val req = Request.Builder()
                    .url("http://$host:$port/api/v1/sync/clock")
                    .header("Authorization", "Bearer $token")
                    .header("X-Device-Id", deviceId)
                    .post(json.encodeToString(reqPayload).toRequestBody(jsonMedia))
                    .build()
                http.newCall(req).execute().use { resp ->
                    val t4 = SystemClock.elapsedRealtimeNanos()
                    if (!resp.isSuccessful) error("Clock sync failed HTTP ${resp.code}")
                    val syncResp = json.decodeFromString(ClockSyncResponse.serializer(), resp.body!!.string())
                    val sample = ClockSyncReportPayload(seq, t1, syncResp.serverT2Ns, syncResp.serverT3Ns, t4)
                    samples.add(sample)
                    // Report back to Studio for linear drift model fitting
                    val repReq = Request.Builder()
                        .url("http://$host:$port/api/v1/sync/report")
                        .header("Authorization", "Bearer $token")
                        .header("X-Device-Id", deviceId)
                        .post(json.encodeToString(sample).toRequestBody(jsonMedia))
                        .build()
                    http.newCall(repReq).execute().close()
                }
                delay(40)
            }
            ClockSyncMath.summarize(samples)
        }
    }

    /**
     * Resumable chunked upload with strict SHA-256 end-to-end verification.
     */
    suspend fun uploadFileResumable(
        host: String,
        port: Int,
        deviceId: String,
        token: String,
        sessionId: String,
        takeId: String,
        file: File,
        fileKind: String,
        sha256Hex: String,
        onProgress: (uploadedBytes: Long, totalBytes: Long) -> Unit
    ): Result<TransferStatusResponse> = withContext(Dispatchers.IO) {
        runCatching {
            val totalSize = file.length()
            val initReq = TransferInitRequest(
                deviceId = deviceId,
                sessionId = sessionId,
                takeId = takeId,
                filename = file.name,
                fileKind = fileKind,
                totalSizeBytes = totalSize,
                sha256Hex = sha256Hex
            )
            val initCall = Request.Builder()
                .url("http://$host:$port/api/v1/transfers/init")
                .header("Authorization", "Bearer $token")
                .header("X-Device-Id", deviceId)
                .post(json.encodeToString(initReq).toRequestBody(jsonMedia))
                .build()

            val initResp = http.newCall(initCall).execute().use { resp ->
                if (!resp.isSuccessful) error("Transfer init failed HTTP ${resp.code}")
                json.decodeFromString(TransferInitResponse.serializer(), resp.body!!.string())
            }

            if (!initResp.alreadyComplete) {
                var offset = initResp.receivedBytes
                val chunkSize = initResp.chunkSizeBytes.coerceIn(64 * 1024, 2 * 1024 * 1024)
                val buf = ByteArray(chunkSize)
                RandomAccessFile(file, "r").use { raf ->
                    raf.seek(offset)
                    while (offset < totalSize) {
                        val read = raf.read(buf)
                        if (read <= 0) break
                        val slice = if (read == buf.size) buf else buf.copyOf(read)
                        val chunkReq = Request.Builder()
                            .url("http://$host:$port/api/v1/transfers/${initResp.transferId}/chunk")
                            .header("Authorization", "Bearer $token")
                            .header("X-Device-Id", deviceId)
                            .header("X-Chunk-Offset", offset.toString())
                            .put(slice.toRequestBody(octetMedia))
                            .build()
                        http.newCall(chunkReq).execute().use { r ->
                            if (!r.isSuccessful) error("Chunk upload failed HTTP ${r.code} at offset $offset")
                        }
                        offset += read
                        onProgress(offset, totalSize)
                    }
                }
            }

            val finReq = Request.Builder()
                .url("http://$host:$port/api/v1/transfers/${initResp.transferId}/finalize")
                .header("Authorization", "Bearer $token")
                .header("X-Device-Id", deviceId)
                .post(ByteArray(0).toRequestBody(octetMedia))
                .build()

            http.newCall(finReq).execute().use { resp ->
                if (!resp.isSuccessful) error("Transfer finalize failed HTTP ${resp.code}")
                json.decodeFromString(TransferStatusResponse.serializer(), resp.body!!.string())
            }
        }
    }
}
