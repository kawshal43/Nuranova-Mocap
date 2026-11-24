package lk.nuranova.mocapcamera.network

import android.content.Context
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import android.net.wifi.WifiManager
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
import lk.nuranova.mocapcamera.protocol.StudioBeaconMessage
import lk.nuranova.mocapcamera.protocol.TransferInitRequest
import lk.nuranova.mocapcamera.protocol.TransferInitResponse
import lk.nuranova.mocapcamera.protocol.TransferStatusResponse
import lk.nuranova.mocapcamera.protocol.UDP_DISCOVERY_PORT
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
import java.net.ConnectException
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.SocketTimeoutException
import java.net.UnknownHostException
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

data class DiscoveredStudio(
    val serviceName: String,
    val host: String,
    val port: Int,
    val source: String = "LAN"
)

data class HealthCheckResult(
    val ok: Boolean,
    val studioName: String = "",
    val message: String = ""
)

/**
 * Local-LAN client handling:
 * - Dual discovery: MulticastLock-enabled Android NSD (mDNS) + UDP LAN Broadcast Beacon (port 8766)
 * - Friendly health check diagnostics (firewall vs port-closed vs wrong subnet)
 * - PIN/QR pairing, authenticated WebSocket control with close-code reporting
 * - Repeated NTP clock synchronization, telemetry heartbeats, and resumable SHA-256 uploads
 */
class StudioClient(private val context: Context) {

    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }
    private val jsonMedia = "application/json; charset=utf-8".toMediaType()
    private val octetMedia = "application/octet-stream".toMediaType()

    private val http = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(25, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .pingInterval(15, TimeUnit.SECONDS)
        .build()

    private var webSocket: WebSocket? = null
    private var nsdManager: NsdManager? = null
    private var discoveryListener: NsdManager.DiscoveryListener? = null
    private var multicastLock: WifiManager.MulticastLock? = null
    private val resolveQueue = ConcurrentLinkedQueue<NsdServiceInfo>()
    private val resolving = AtomicBoolean(false)

    private val udpRunning = AtomicBoolean(false)
    private var udpThread: Thread? = null

    var onScheduleRecord: ((ScheduleRecordCommand) -> Unit)? = null
    var onStopRecord: ((StopRecordCommand) -> Unit)? = null
    var onConnectionStateChanged: ((connected: Boolean, message: String, closeCode: Int?) -> Unit)? = null

    @Suppress("DEPRECATION")
    fun startDiscovery(onStudioFound: (DiscoveredStudio) -> Unit) {
        stopDiscovery()

        // 1. Acquire MulticastLock so Samsung/Xiaomi Wi-Fi chips receive mDNS & LAN broadcasts
        runCatching {
            val wm = context.applicationContext.getSystemService(Context.WIFI_SERVICE) as? WifiManager
            multicastLock = wm?.createMulticastLock("nuranova_mdns_lock")?.apply {
                setReferenceCounted(false)
                acquire()
            }
        }

        // 2. Start Android NSD (mDNS) discovery with serialized resolve queue
        val mgr = context.getSystemService(Context.NSD_SERVICE) as? NsdManager
        if (mgr != null) {
            nsdManager = mgr
            val listener = object : NsdManager.DiscoveryListener {
                override fun onDiscoveryStarted(regType: String) {}
                override fun onServiceFound(serviceInfo: NsdServiceInfo) {
                    if (serviceInfo.serviceType.contains("nuranova-mocap")) {
                        resolveQueue.add(serviceInfo)
                        drainResolveQueue(mgr, onStudioFound)
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

        // 3. Start UDP Discovery Beacon listener + active probe (works even when routers filter mDNS!)
        startUdpDiscovery(onStudioFound)
    }

    @Suppress("DEPRECATION")
    private fun drainResolveQueue(mgr: NsdManager, onStudioFound: (DiscoveredStudio) -> Unit) {
        if (!resolving.compareAndSet(false, true)) return
        val next = resolveQueue.poll()
        if (next == null) {
            resolving.set(false)
            return
        }
        runCatching {
            mgr.resolveService(next, object : NsdManager.ResolveListener {
                override fun onResolveFailed(info: NsdServiceInfo, errorCode: Int) {
                    resolving.set(false)
                    drainResolveQueue(mgr, onStudioFound)
                }

                override fun onServiceResolved(info: NsdServiceInfo) {
                    val ip = info.host?.hostAddress
                    if (!ip.isNullOrBlank() && ip != "127.0.0.1") {
                        onStudioFound(DiscoveredStudio(info.serviceName, ip, info.port, source = "mDNS"))
                    }
                    resolving.set(false)
                    drainResolveQueue(mgr, onStudioFound)
                }
            })
        }.onFailure {
            resolving.set(false)
        }
    }

    private fun startUdpDiscovery(onStudioFound: (DiscoveredStudio) -> Unit) {
        udpRunning.set(true)
        udpThread = Thread({
            var sock: DatagramSocket? = null
            try {
                sock = DatagramSocket(null).apply {
                    reuseAddress = true
                    broadcast = true
                    try {
                        bind(InetSocketAddress(UDP_DISCOVERY_PORT))
                    } catch (_: Exception) {
                        bind(InetSocketAddress(0))
                    }
                    soTimeout = 450
                }
                val probeBytes = """{"type":"nuranova_discover"}""".toByteArray(Charsets.UTF_8)
                val buf = ByteArray(1024)
                var lastProbeMs = 0L

                while (udpRunning.get()) {
                    val now = SystemClock.elapsedRealtime()
                    if (now - lastProbeMs >= 1800L) {
                        lastProbeMs = now
                        runCatching {
                            val bcast = InetAddress.getByName("255.255.255.255")
                            sock.send(DatagramPacket(probeBytes, probeBytes.size, bcast, UDP_DISCOVERY_PORT))
                        }
                    }
                    try {
                        val pkt = DatagramPacket(buf, buf.size)
                        sock.receive(pkt)
                        val text = String(pkt.data, pkt.offset, pkt.length, Charsets.UTF_8)
                        val beacon = runCatching { json.decodeFromString(StudioBeaconMessage.serializer(), text) }.getOrNull()
                        if (beacon != null && beacon.type == "nuranova_studio_beacon") {
                            val host = beacon.host.ifBlank { pkt.address?.hostAddress.orEmpty() }
                            if (host.isNotBlank() && host != "127.0.0.1") {
                                onStudioFound(
                                    DiscoveredStudio(
                                        serviceName = beacon.studioName,
                                        host = host,
                                        port = beacon.port,
                                        source = "UDP"
                                    )
                                )
                            }
                        }
                    } catch (_: SocketTimeoutException) {
                        // normal poll cycle
                    }
                }
            } catch (_: Exception) {
                // UDP discovery unavailable on this interface
            } finally {
                runCatching { sock?.close() }
            }
        }, "nuranova-udp-discover").also { it.isDaemon = true; it.start() }
    }

    fun stopDiscovery() {
        udpRunning.set(false)
        runCatching { udpThread?.interrupt() }
        udpThread = null
        resolveQueue.clear()
        resolving.set(false)
        val mgr = nsdManager
        val listener = discoveryListener
        if (mgr != null && listener != null) {
            runCatching { mgr.stopServiceDiscovery(listener) }
        }
        discoveryListener = null
        runCatching {
            if (multicastLock?.isHeld == true) multicastLock?.release()
        }
        multicastLock = null
    }

    private fun friendlyNetworkError(err: Throwable, host: String, port: Int): String = when (err) {
        is SocketTimeoutException ->
            "Timed out reaching $host:$port. Check that phone & PC share the same Wi-Fi/LAN and click 'Fix Windows Firewall' in Studio."
        is ConnectException ->
            "Connection refused at $host:$port. Make sure Nuranova Mocap Studio Server is ONLINE on your PC."
        is UnknownHostException ->
            "Invalid PC address '$host'. Check the IP shown in Studio → Cameras & Pairing."
        else -> err.message ?: "Network error contacting $host:$port"
    }

    suspend fun checkHealth(host: String, port: Int): HealthCheckResult = withContext(Dispatchers.IO) {
        val h = host.trim()
        if (h.isBlank()) return@withContext HealthCheckResult(false, message = "Enter your Studio PC IP address first.")
        try {
            val req = Request.Builder().url("http://$h:$port/api/v1/health").get().build()
            http.newCall(req).execute().use { resp ->
                val body = resp.body?.string().orEmpty()
                if (!resp.isSuccessful) {
                    HealthCheckResult(false, message = "Studio returned HTTP ${resp.code}: $body")
                } else {
                    val obj = runCatching { json.parseToJsonElement(body).jsonObject }.getOrNull()
                    val name = obj?.get("studio_name")?.jsonPrimitive?.content ?: "Nuranova Mocap Studio"
                    HealthCheckResult(true, studioName = name, message = "Reachable! Found $name at $h:$port")
                }
            }
        } catch (e: Throwable) {
            HealthCheckResult(false, message = friendlyNetworkError(e, h, port))
        }
    }

    suspend fun pair(host: String, port: Int, pairReq: PairRequest): Result<PairResponse> =
        withContext(Dispatchers.IO) {
            val h = host.trim()
            try {
                val body = json.encodeToString(pairReq).toRequestBody(jsonMedia)
                val req = Request.Builder().url("http://$h:$port/api/v1/pair").post(body).build()
                http.newCall(req).execute().use { resp ->
                    val text = resp.body?.string().orEmpty()
                    if (!resp.isSuccessful) {
                        val detail = runCatching {
                            json.parseToJsonElement(text).jsonObject["detail"]?.jsonPrimitive?.content
                        }.getOrNull()
                        val msg = when (resp.code) {
                            403 -> detail ?: "Invalid or expired 6-digit PIN. Check the PIN shown in Mocap Studio."
                            400 -> detail ?: "Pairing rejected (HTTP 400)."
                            else -> "Pairing failed (HTTP ${resp.code}): ${detail ?: text}"
                        }
                        Result.failure(IllegalStateException(msg))
                    } else {
                        Result.success(json.decodeFromString(PairResponse.serializer(), text))
                    }
                }
            } catch (e: Throwable) {
                Result.failure(IllegalStateException(friendlyNetworkError(e, h, port), e))
            }
        }

    fun connectWebSocket(host: String, port: Int, deviceId: String, authToken: String) {
        disconnectWebSocket()
        val req = Request.Builder()
            .url("ws://${host.trim()}:$port/ws/v1/camera/$deviceId?token=$authToken")
            .header("Authorization", "Bearer $authToken")
            .build()

        webSocket = http.newWebSocket(req, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                onConnectionStateChanged?.invoke(true, "Connected to ${host.trim()}:$port", null)
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
                val code = response?.code
                val msg = if (code == 401 || code == 403) {
                    "Authentication expired — re-pair with Studio QR or PIN"
                } else {
                    friendlyNetworkError(t, host, port)
                }
                onConnectionStateChanged?.invoke(false, msg, code)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                val msg = if (code == 4401) {
                    "Unpaired by Studio (4401) — scan QR or enter PIN to re-pair"
                } else {
                    "Disconnected ($code)"
                }
                onConnectionStateChanged?.invoke(false, msg, code)
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
                    .url("http://${host.trim()}:$port/api/v1/telemetry")
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
            val h = host.trim()
            val samples = mutableListOf<ClockSyncReportPayload>()
            for (seq in 0 until probes) {
                val t1 = SystemClock.elapsedRealtimeNanos()
                val reqPayload = ClockSyncRequest(deviceId = deviceId, seq = seq, clientT1Ns = t1)
                val req = Request.Builder()
                    .url("http://$h:$port/api/v1/sync/clock")
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
                    val repReq = Request.Builder()
                        .url("http://$h:$port/api/v1/sync/report")
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
            val h = host.trim()
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
                .url("http://$h:$port/api/v1/transfers/init")
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
                            .url("http://$h:$port/api/v1/transfers/${initResp.transferId}/chunk")
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
                .url("http://$h:$port/api/v1/transfers/${initResp.transferId}/finalize")
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
