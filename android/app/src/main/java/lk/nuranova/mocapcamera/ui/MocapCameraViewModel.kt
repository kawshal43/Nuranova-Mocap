package lk.nuranova.mocapcamera.ui

import android.app.Application
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.StatFs
import android.os.SystemClock
import android.view.Surface
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import lk.nuranova.mocapcamera.camera.Camera2Controller
import lk.nuranova.mocapcamera.data.LocalRecordingEntity
import lk.nuranova.mocapcamera.data.MocapDatabase
import lk.nuranova.mocapcamera.data.SettingsStore
import lk.nuranova.mocapcamera.data.StoredAppConfig
import lk.nuranova.mocapcamera.network.DiscoveredStudio
import lk.nuranova.mocapcamera.network.StudioClient
import lk.nuranova.mocapcamera.protocol.CameraCapabilities
import lk.nuranova.mocapcamera.protocol.ClockSyncSummary
import lk.nuranova.mocapcamera.protocol.DeviceTelemetry
import lk.nuranova.mocapcamera.protocol.PairRequest
import lk.nuranova.mocapcamera.protocol.RecordingConfig
import lk.nuranova.mocapcamera.service.RecordingForegroundService
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class MocapUiState(
    val config: StoredAppConfig = StoredAppConfig(),
    val capabilities: CameraCapabilities = CameraCapabilities(),
    val recordingConfig: RecordingConfig = RecordingConfig(manualExposure = false),
    val discoveredStudios: List<DiscoveredStudio> = emptyList(),
    val connected: Boolean = false,
    val connectionMessage: String = "Disconnected",
    val isRecording: Boolean = false,
    val activeTakeId: String? = null,
    val recordingElapsedSeconds: Double = 0.0,
    val measuredFps: Double = 0.0,
    val droppedFrames: Int = 0,
    val currentShutterNs: Long? = 4_000_000L,
    val currentIso: Int? = 400,
    val syncSummary: ClockSyncSummary? = null,
    val batteryPercent: Double = 100.0,
    val temperatureC: Double? = null,
    val freeStorageMb: Double = 0.0,
    val lastError: String? = null,
    val statusLog: List<String> = emptyList()
)

class MocapCameraViewModel(application: Application) : AndroidViewModel(application) {

    private val ctx = application.applicationContext
    private val settingsStore = SettingsStore(ctx)
    private val db = MocapDatabase.getInstance(ctx)
    private val recordingDao = db.recordingDao()
    val cameraController = Camera2Controller(ctx)
    private val studioClient = StudioClient(ctx)

    private val _uiState = MutableStateFlow(MocapUiState())
    val uiState: StateFlow<MocapUiState> = _uiState.asStateFlow()

    val localRecordings: StateFlow<List<LocalRecordingEntity>> = recordingDao.observeAll()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    private var activePreviewSurface: Surface? = null
    private var recordingStartMs: Long = 0L

    init {
        viewModelScope.launch {
            settingsStore.ensureDeviceId()
            settingsStore.configFlow.collect { cfg ->
                _uiState.update { it.copy(config = cfg) }
            }
        }

        cameraController.onPreviewJpegReady = { jpeg ->
            if (_uiState.value.connected) studioClient.sendPreviewFrame(jpeg)
        }
        cameraController.onMetricsUpdated = { fps, dropped, shutter, iso ->
            _uiState.update {
                it.copy(measuredFps = fps, droppedFrames = dropped, currentShutterNs = shutter, currentIso = iso)
            }
        }
        cameraController.onWarning = { warn ->
            appendLog("Warning: $warn")
        }
        cameraController.onError = { err ->
            appendLog("Camera Error: $err")
            _uiState.update { it.copy(lastError = err) }
        }

        studioClient.onConnectionStateChanged = { ok, msg ->
            appendLog("Network: $msg")
            _uiState.update { it.copy(connected = ok, connectionMessage = msg) }
        }
        studioClient.onScheduleRecord = { cmd ->
            viewModelScope.launch {
                val nowNs = SystemClock.elapsedRealtimeNanos()
                val waitMs = ((cmd.scheduledStartDeviceNs - nowNs) / 1_000_000L).coerceIn(0L, 5000L)
                appendLog("Scheduled take '${cmd.takeId}' in ${waitMs} ms")
                if (waitMs > 0L) delay(waitMs)
                startRecordingInternal(cmd.sessionId, cmd.takeId, cmd.config)
            }
        }
        studioClient.onStopRecord = { cmd ->
            viewModelScope.launch {
                appendLog("Remote STOP received for '${cmd.takeId}'")
                stopRecordingAndQueueUpload()
            }
        }

        // Periodic device health monitor & telemetry loop
        viewModelScope.launch {
            while (true) {
                refreshHardwareHealth()
                val st = _uiState.value
                if (st.isRecording) {
                    val elapsed = (SystemClock.elapsedRealtime() - recordingStartMs) / 1000.0
                    _uiState.update { it.copy(recordingElapsedSeconds = elapsed) }
                }
                if (st.connected && st.config.authToken.isNotBlank()) {
                    studioClient.postTelemetry(
                        st.config.studioHost,
                        st.config.studioPort,
                        st.config.authToken,
                        DeviceTelemetry(
                            deviceId = st.config.deviceId,
                            batteryPercent = st.batteryPercent,
                            temperatureCelsius = st.temperatureC,
                            availableStorageMb = st.freeStorageMb,
                            actualWidth = st.recordingConfig.width,
                            actualHeight = st.recordingConfig.height,
                            measuredFps = st.measuredFps,
                            droppedFrames = st.droppedFrames,
                            shutterSpeedNs = st.currentShutterNs,
                            iso = st.currentIso,
                            focusLocked = st.recordingConfig.lockFocus,
                            wbLocked = st.recordingConfig.lockWhiteBalance,
                            recordingState = if (st.isRecording) "recording" else "idle",
                            activeTakeId = st.activeTakeId,
                            recordingDurationS = st.recordingElapsedSeconds,
                            monotonicTimeNs = SystemClock.elapsedRealtimeNanos()
                        )
                    )
                }
                delay(1500)
            }
        }
    }

    fun probeCameraCapabilities() {
        runCatching {
            val caps = cameraController.probeCapabilities(_uiState.value.recordingConfig.cameraLens)
            _uiState.update { it.copy(capabilities = caps) }
            appendLog("Probed Camera2 (${caps.hardwareLevel}, TS=${caps.timestampSource}, ${caps.supportedModes.size} modes)")
        }.onFailure {
            _uiState.update { s -> s.copy(lastError = "Camera probe failed: ${it.message}") }
        }
    }

    fun bindPreviewSurface(surface: Surface?) {
        activePreviewSurface = surface
        if (surface != null) {
            runCatching { cameraController.openCamera(surface, _uiState.value.recordingConfig) }
                .onFailure { _uiState.update { s -> s.copy(lastError = "Camera open failed: ${it.message}") } }
        }
    }

    fun startDiscovery() {
        _uiState.update { it.copy(discoveredStudios = emptyList()) }
        studioClient.startDiscovery { studio ->
            _uiState.update { st ->
                val list = st.discoveredStudios.filterNot { it.host == studio.host && it.port == studio.port } + studio
                st.copy(discoveredStudios = list)
            }
        }
    }

    fun stopDiscovery() = studioClient.stopDiscovery()

    fun pairWithStudio(host: String, port: Int, pin: String, role: String = "body") {
        viewModelScope.launch {
            val st = _uiState.value
            val caps = runCatching { cameraController.probeCapabilities(st.recordingConfig.cameraLens) }
                .getOrDefault(st.capabilities)
            val req = PairRequest(
                deviceId = st.config.deviceId,
                deviceName = st.config.deviceName,
                pin = pin.trim(),
                capabilities = caps,
                role = role
            )
            studioClient.pair(host.trim(), port, req)
                .onSuccess { resp ->
                    settingsStore.savePairing(host.trim(), port, resp.authToken, resp.assignedCameraUid, role)
                    appendLog("Paired as ${resp.assignedCameraUid} with ${resp.studioName}")
                    studioClient.connectWebSocket(host.trim(), port, resp.deviceId, resp.authToken)
                    runClockSync()
                }
                .onFailure { err ->
                    appendLog("Pairing failed: ${err.message}")
                    _uiState.update { it.copy(lastError = err.message) }
                }
        }
    }

    fun reconnectStudio() {
        val cfg = _uiState.value.config
        if (cfg.studioHost.isNotBlank() && cfg.authToken.isNotBlank()) {
            studioClient.connectWebSocket(cfg.studioHost, cfg.studioPort, cfg.deviceId, cfg.authToken)
            runClockSync()
        }
    }

    fun runClockSync() {
        viewModelScope.launch {
            val cfg = _uiState.value.config
            if (cfg.studioHost.isBlank() || cfg.authToken.isBlank()) return@launch
            studioClient.runClockSyncBurst(cfg.studioHost, cfg.studioPort, cfg.deviceId, cfg.authToken)
                .onSuccess { summary ->
                    appendLog("Clock Sync: residual=%.2f ms, minRTT=%.2f ms".format(summary.residualMs, summary.minRttMs))
                    _uiState.update { it.copy(syncSummary = summary) }
                }
                .onFailure { appendLog("Clock Sync failed: ${it.message}") }
        }
    }

    fun updateRecordingConfig(newConfig: RecordingConfig) {
        _uiState.update { it.copy(recordingConfig = newConfig) }
        activePreviewSurface?.let { bindPreviewSurface(it) }
    }

    fun toggleLocalRecording() {
        if (_uiState.value.isRecording) {
            viewModelScope.launch { stopRecordingAndQueueUpload() }
        } else {
            val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            startRecordingInternal("local_session", "take_$stamp", _uiState.value.recordingConfig)
        }
    }

    private fun startRecordingInternal(sessionId: String, takeId: String, cfg: RecordingConfig) {
        val outDir = File(ctx.getExternalFilesDir(null) ?: ctx.filesDir, "takes/$takeId")
        val svcIntent = Intent(ctx, RecordingForegroundService::class.java)
            .putExtra(RecordingForegroundService.EXTRA_STATUS, "Recording $takeId (${cfg.width}x${cfg.height} @ ${cfg.targetFps.toInt()}fps)")
        runCatching { ctx.startForegroundService(svcIntent) }

        recordingStartMs = SystemClock.elapsedRealtime()
        cameraController.startRecording(sessionId, takeId, cfg, activePreviewSurface, outDir)
        _uiState.update { it.copy(isRecording = true, activeTakeId = takeId, recordingElapsedSeconds = 0.0) }
        appendLog("Recording STARTED: $takeId")
    }

    private suspend fun stopRecordingAndQueueUpload() {
        val result = cameraController.stopRecording(activePreviewSurface)
        runCatching { ctx.stopService(Intent(ctx, RecordingForegroundService::class.java)) }
        _uiState.update { it.copy(isRecording = false, activeTakeId = null) }

        if (result != null) {
            appendLog("Recorded ${result.takeId}: ${result.frameCount} frames (%.2f fps)".format(result.measuredFps))
            val entity = LocalRecordingEntity(
                takeId = result.takeId,
                sessionId = result.sessionId,
                videoPath = result.videoFile.absolutePath,
                timestampPath = result.timestampFile.absolutePath,
                videoSizeBytes = result.videoFile.length(),
                videoSha256 = result.sha256Hex,
                frameCount = result.frameCount,
                width = result.width,
                height = result.height,
                measuredFps = result.measuredFps,
                transferStatus = "pending"
            )
            recordingDao.upsert(entity)
            if (_uiState.value.config.autoUploadAfterRecord && _uiState.value.connected) {
                uploadRecording(entity)
            }
        }
    }

    fun uploadRecording(entity: LocalRecordingEntity) {
        viewModelScope.launch {
            val cfg = _uiState.value.config
            if (cfg.studioHost.isBlank() || cfg.authToken.isBlank()) {
                appendLog("Cannot upload '${entity.takeId}': not paired to Studio.")
                return@launch
            }
            val videoFile = File(entity.videoPath)
            val tsFile = File(entity.timestampPath)
            if (!videoFile.exists()) {
                recordingDao.updateTransfer(entity.takeId, "failed", 0L, false)
                return@launch
            }
            recordingDao.updateTransfer(entity.takeId, "uploading", entity.bytesUploaded, false)
            studioClient.uploadFileResumable(
                host = cfg.studioHost,
                port = cfg.studioPort,
                deviceId = cfg.deviceId,
                token = cfg.authToken,
                sessionId = entity.sessionId,
                takeId = entity.takeId,
                file = videoFile,
                fileKind = "video",
                sha256Hex = entity.videoSha256
            ) { uploaded, _ ->
                viewModelScope.launch {
                    recordingDao.updateTransfer(entity.takeId, "uploading", uploaded, false)
                }
            }.onSuccess { status ->
                if (tsFile.exists()) {
                    studioClient.uploadFileResumable(
                        cfg.studioHost, cfg.studioPort, cfg.deviceId, cfg.authToken,
                        entity.sessionId, entity.takeId, tsFile, "timestamps",
                        Camera2Controller.computeSha256(tsFile)
                    ) { _, _ -> }
                }
                recordingDao.updateTransfer(
                    entity.takeId,
                    if (status.verified) "verified" else "failed",
                    status.receivedBytes,
                    status.verified
                )
                appendLog("Transfer '${entity.takeId}' SHA-256 verified=${status.verified}")
                // Never delete unless Studio explicitly verified SHA-256 AND user disabled local retention
                if (status.canDeleteLocalCopy && !cfg.retainLocalCopyAfterVerify) {
                    videoFile.delete()
                }
            }.onFailure { err ->
                recordingDao.updateTransfer(entity.takeId, "failed", entity.bytesUploaded, false)
                appendLog("Transfer failed '${entity.takeId}': ${err.message}")
            }
        }
    }

    fun saveAppSettings(name: String, autoUpload: Boolean, retainLocal: Boolean) {
        viewModelScope.launch { settingsStore.updateSettings(name, autoUpload, retainLocal) }
    }

    fun completeOnboarding() {
        viewModelScope.launch { settingsStore.markOnboardingComplete() }
    }

    private fun refreshHardwareHealth() {
        val battIntent = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val level = battIntent?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = battIntent?.getIntExtra(BatteryManager.EXTRA_SCALE, 100) ?: 100
        val tempTenths = battIntent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE) ?: Int.MIN_VALUE
        val pct = if (level >= 0 && scale > 0) (level * 100.0 / scale) else 100.0
        val tempC = if (tempTenths != Int.MIN_VALUE) tempTenths / 10.0 else null

        val dir = ctx.getExternalFilesDir(null) ?: ctx.filesDir
        val stat = StatFs(dir.absolutePath)
        val freeMb = stat.availableBytes / (1024.0 * 1024.0)
        _uiState.update { it.copy(batteryPercent = pct, temperatureC = tempC, freeStorageMb = freeMb) }
    }

    private fun appendLog(msg: String) {
        val ts = SimpleDateFormat("HH:mm:ss", Locale.US).format(Date())
        _uiState.update { st -> st.copy(statusLog = (st.statusLog + "[$ts] $msg").takeLast(60)) }
    }

    override fun onCleared() {
        super.onCleared()
        studioClient.stopDiscovery()
        studioClient.disconnectWebSocket()
        cameraController.close()
    }
}
