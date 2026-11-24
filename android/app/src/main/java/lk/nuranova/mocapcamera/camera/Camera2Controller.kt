package lk.nuranova.mocapcamera.camera

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.media.Image
import android.media.ImageReader
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Range
import android.util.Size
import android.view.Surface
import com.google.zxing.BarcodeFormat
import com.google.zxing.BinaryBitmap
import com.google.zxing.DecodeHintType
import com.google.zxing.MultiFormatReader
import com.google.zxing.PlanarYUVLuminanceSource
import com.google.zxing.common.HybridBinarizer
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import lk.nuranova.mocapcamera.protocol.CameraCapabilities
import lk.nuranova.mocapcamera.protocol.FrameTimestampRecord
import lk.nuranova.mocapcamera.protocol.RecordingConfig
import lk.nuranova.mocapcamera.protocol.SupportedMode
import java.io.BufferedWriter
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileWriter
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

data class CompletedRecordingResult(
    val takeId: String,
    val sessionId: String,
    val videoFile: File,
    val timestampFile: File,
    val sha256Hex: String,
    val frameCount: Int,
    val width: Int,
    val height: Int,
    val measuredFps: Double
)

/**
 * Dynamic Camera2 API controller supporting:
 * - Runtime hardware capability inspection (never hardcoded for specific phones)
 * - Locked manual exposure/focus/WB & disabled digital stabilization where supported
 * - High-bitrate local H.264 MediaCodec + MediaMuxer recording
 * - Low-bandwidth JPEG preview extraction for desktop monitoring
 * - Inline ZXing QR code decoding from live YUV frames for 1-tap Studio pairing
 * - Per-frame Camera2 SENSOR_TIMESTAMP + exposure + rolling shutter skew logging
 */
class Camera2Controller(private val context: Context) {

    private val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private val json = Json { encodeDefaults = true }
    private val qrReader = MultiFormatReader().apply {
        setHints(
            mapOf(
                DecodeHintType.POSSIBLE_FORMATS to listOf(BarcodeFormat.QR_CODE),
                DecodeHintType.TRY_HARDER to true
            )
        )
    }

    private var bgThread: HandlerThread? = null
    private var bgHandler: Handler? = null
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var previewImageReader: ImageReader? = null

    private var mediaCodec: MediaCodec? = null
    private var mediaMuxer: MediaMuxer? = null
    private var encoderSurface: Surface? = null
    private var encoderDrainThread: Thread? = null
    private var muxerTrackIndex: Int = -1
    private var muxerStarted: Boolean = false

    private var timestampWriter: BufferedWriter? = null
    private var currentVideoFile: File? = null
    private var currentTimestampFile: File? = null
    private var activeSessionId: String = ""
    private var activeTakeId: String = ""
    private var activeConfig: RecordingConfig = RecordingConfig()
    @Volatile
    private var negotiatedMaxFps: Double = 30.0

    private val isRecording = AtomicBoolean(false)
    val qrScanEnabled = AtomicBoolean(false)
    private val frameCounter = AtomicInteger(0)
    private val previewCallbackCounter = AtomicInteger(0)
    private val droppedFrameCounter = AtomicInteger(0)
    private val lastFrameSensorNs = AtomicLong(0L)
    private val lastPreviewSentMs = AtomicLong(0L)
    private val lastQrScanMs = AtomicLong(0L)
    private var recordStartElapsedMs: Long = 0L

    var onPreviewJpegReady: ((ByteArray) -> Unit)? = null
    var onQrCodeDetected: ((String) -> Unit)? = null
    var onMetricsUpdated: ((fps: Double, dropped: Int, shutterNs: Long?, iso: Int?) -> Unit)? = null
    var onWarning: ((String) -> Unit)? = null
    var onError: ((String) -> Unit)? = null

    fun chooseSupportedPreviewSize(lensPreference: String = "rear_main", targetW: Int = 1920, targetH: Int = 1080): Size {
        val camId = selectCameraId(lensPreference)
        val map = cameraManager.getCameraCharacteristics(camId)
            .get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return Size(1280, 720)
        val sizes = map.getOutputSizes(android.graphics.SurfaceTexture::class.java)?.toList() ?: return Size(1280, 720)
        return sizes.minByOrNull {
            kotlin.math.abs(it.width - targetW) + kotlin.math.abs(it.height - targetH)
        } ?: Size(1280, 720)
    }

    private fun chooseSupportedYuvPreviewSize(cameraId: String, targetW: Int, targetH: Int): Size {
        val map = cameraManager.getCameraCharacteristics(cameraId)
            .get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return Size(640, 480)
        val sizes = map.getOutputSizes(ImageFormat.YUV_420_888)?.toList() ?: return Size(640, 480)
        // 640x480 / 640x360 works cleanly for both crisp QR decoding and low-bandwidth preview streaming
        val candidates = sizes.filter { it.width >= 480 && it.height >= 270 && it.width <= 960 }
        val pool = if (candidates.isNotEmpty()) candidates else sizes
        return pool.minByOrNull {
            kotlin.math.abs(it.width - targetW) + kotlin.math.abs(it.height - targetH)
        } ?: Size(640, 480)
    }

    private fun chooseSupportedFpsRange(cameraId: String, requestedFps: Int): Pair<Range<Int>, Boolean> {
        val chars = cameraManager.getCameraCharacteristics(cameraId)
        val ranges = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)?.toList()
            ?: return Range(30, 30) to false
        ranges.firstOrNull { it.lower == requestedFps && it.upper == requestedFps }?.let { return it to true }
        val fixed = ranges.filter { it.lower == it.upper }
        val bestFixed = fixed.filter { it.upper <= requestedFps }.maxByOrNull { it.upper }
            ?: fixed.maxByOrNull { it.upper }
        if (bestFixed != null) {
            return bestFixed to (bestFixed.upper == requestedFps)
        }
        val best = ranges.maxByOrNull { it.upper } ?: Range(30, 30)
        return best to (best.upper == requestedFps)
    }

    fun probeCapabilities(lensPreference: String = "rear_main"): CameraCapabilities {
        val camId = selectCameraId(lensPreference)
        val chars = cameraManager.getCameraCharacteristics(camId)

        val hwLevel = when (chars.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)) {
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_3 -> "LEVEL_3"
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL -> "FULL"
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED -> "LIMITED"
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_EXTERNAL -> "EXTERNAL"
            else -> "LEGACY"
        }

        val tsSource = when (chars.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)) {
            CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME -> "REALTIME"
            CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_UNKNOWN -> "MONOTONIC"
            else -> "UNKNOWN"
        }

        val facing = when (chars.get(CameraCharacteristics.LENS_FACING)) {
            CameraCharacteristics.LENS_FACING_FRONT -> "front"
            CameraCharacteristics.LENS_FACING_EXTERNAL -> "external"
            else -> "back"
        }

        val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        val aeRanges = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)?.toList() ?: emptyList()
        val fpsRanges = aeRanges.map { listOf(it.lower, it.upper) }
        val maxAeFps = aeRanges.maxOfOrNull { it.upper }?.toDouble() ?: 30.0

        val standardTargets = listOf(
            Size(1280, 720),
            Size(1920, 1080),
            Size(2560, 1440),
            Size(3840, 2160)
        )
        val modes = mutableListOf<SupportedMode>()
        if (map != null) {
            val outSizes = map.getOutputSizes(MediaCodec::class.java)?.toList() ?: emptyList()
            for (sz in standardTargets) {
                if (outSizes.any { it.width == sz.width && it.height == sz.height }) {
                    val minDurNs = runCatching { map.getOutputMinFrameDuration(MediaCodec::class.java, sz) }.getOrDefault(0L)
                    val durFps = if (minDurNs > 0L) (1e9 / minDurNs) else maxAeFps
                    val effectiveMaxFps = minOf(durFps, maxAeFps).coerceAtLeast(15.0)
                    modes.add(
                        SupportedMode(
                            width = sz.width,
                            height = sz.height,
                            fpsRanges = fpsRanges,
                            maxFps = effectiveMaxFps,
                            supportsHighSpeed = false
                        )
                    )
                }
            }
            runCatching {
                map.highSpeedVideoSizes?.forEach { hsSize ->
                    val hsRanges = map.getHighSpeedVideoFpsRangesFor(hsSize)
                    val maxHs = hsRanges.maxOfOrNull { it.upper }?.toDouble() ?: 60.0
                    modes.add(
                        SupportedMode(
                            width = hsSize.width,
                            height = hsSize.height,
                            fpsRanges = hsRanges.map { listOf(it.lower, it.upper) },
                            maxFps = maxHs,
                            supportsHighSpeed = true
                        )
                    )
                }
            }
        }

        val aeModes = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_MODES) ?: IntArray(0)
        val afModes = chars.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES) ?: IntArray(0)
        val stabModes = chars.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES) ?: IntArray(0)
        val expRange = chars.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        val isoRange = chars.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        val focals = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.toList() ?: emptyList()
        val physSize = chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)?.let { listOf(it.width, it.height) }

        return CameraCapabilities(
            manufacturer = Build.MANUFACTURER ?: "Unknown",
            model = Build.MODEL ?: "Unknown",
            androidApiLevel = Build.VERSION.SDK_INT,
            cameraId = camId,
            lensFacing = facing,
            hardwareLevel = hwLevel,
            timestampSource = tsSource,
            supportedModes = modes,
            supportsManualExposure = aeModes.contains(CameraCharacteristics.CONTROL_AE_MODE_OFF),
            supportsManualFocus = afModes.contains(CameraCharacteristics.CONTROL_AF_MODE_OFF),
            supportsAwbLock = chars.get(CameraCharacteristics.CONTROL_AWB_LOCK_AVAILABLE) == true,
            supportsVideoStabilizationOff = stabModes.isEmpty() || stabModes.contains(CameraCharacteristics.CONTROL_VIDEO_STABILIZATION_MODE_OFF),
            exposureTimeRangeNs = expRange?.let { listOf(it.lower, it.upper) },
            isoRange = isoRange?.let { listOf(it.lower, it.upper) },
            focalLengthsMm = focals,
            sensorPhysicalSizeMm = physSize
        )
    }

    private fun selectCameraId(lensPreference: String): String {
        val targetFacing = if (lensPreference == "front") {
            CameraCharacteristics.LENS_FACING_FRONT
        } else {
            CameraCharacteristics.LENS_FACING_BACK
        }
        val ids = cameraManager.cameraIdList
        return ids.firstOrNull { id ->
            cameraManager.getCameraCharacteristics(id).get(CameraCharacteristics.LENS_FACING) == targetFacing
        } ?: ids.first()
    }

    @SuppressLint("MissingPermission")
    fun openCamera(previewSurface: Surface?, config: RecordingConfig) {
        activeConfig = config
        startBackgroundThread()

        // If camera is already open and we're just rebinding the surface, recreate session cleanly
        if (cameraDevice != null && !isRecording.get()) {
            runCatching { captureSession?.close() }
            captureSession = null
            createCaptureSession(previewSurface, includeEncoder = false)
            return
        }

        val camId = selectCameraId(config.cameraLens)
        val yuvSize = chooseSupportedYuvPreviewSize(camId, 640, 360)

        previewImageReader?.close()
        previewImageReader = ImageReader.newInstance(
            yuvSize.width,
            yuvSize.height,
            ImageFormat.YUV_420_888,
            3
        ).apply {
            setOnImageAvailableListener({ reader ->
                val img = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                try {
                    val nowMs = SystemClock.elapsedRealtime()
                    if (qrScanEnabled.get() && nowMs - lastQrScanMs.get() >= 220L) {
                        lastQrScanMs.set(nowMs)
                        val qrText = tryDecodeQrFromImage(img)
                        if (qrText != null) {
                            qrScanEnabled.set(false)
                            onQrCodeDetected?.invoke(qrText)
                        }
                    }
                    val minGapMs = (1000.0 / config.previewFps.coerceAtLeast(1.0)).toLong()
                    if (nowMs - lastPreviewSentMs.get() >= minGapMs && onPreviewJpegReady != null) {
                        lastPreviewSentMs.set(nowMs)
                        val jpeg = yuvImageToJpeg(img, config.previewJpegQuality)
                        if (jpeg != null) onPreviewJpegReady?.invoke(jpeg)
                    }
                } finally {
                    img.close()
                }
            }, bgHandler)
        }

        cameraManager.openCamera(camId, object : CameraDevice.StateCallback() {
            override fun onOpened(device: CameraDevice) {
                cameraDevice = device
                createCaptureSession(previewSurface, includeEncoder = false)
            }

            override fun onDisconnected(device: CameraDevice) {
                device.close()
                cameraDevice = null
            }

            override fun onError(device: CameraDevice, error: Int) {
                device.close()
                cameraDevice = null
                onError?.invoke("Camera2 device error code: $error")
            }
        }, bgHandler)
    }

    private fun tryDecodeQrFromImage(image: Image): String? {
        return runCatching {
            val plane = image.planes[0]
            val w = image.width
            val h = image.height
            val rowStride = plane.rowStride
            val buf = plane.buffer
            buf.rewind()
            val yBytes = ByteArray(w * h)
            if (rowStride == w) {
                buf.get(yBytes, 0, w * h)
            } else {
                val row = ByteArray(rowStride)
                for (r in 0 until h) {
                    val toRead = minOf(rowStride, buf.remaining())
                    buf.get(row, 0, toRead)
                    System.arraycopy(row, 0, yBytes, r * w, w)
                }
            }
            val src = PlanarYUVLuminanceSource(yBytes, w, h, 0, 0, w, h, false)
            try {
                qrReader.decodeWithState(BinaryBitmap(HybridBinarizer(src))).text
            } catch (_: Exception) {
                // Retry on 90-degree rotated luma (portrait phones scanning desktop monitor QR!)
                val rot = ByteArray(w * h)
                for (y in 0 until h) {
                    for (x in 0 until w) {
                        rot[x * h + (h - 1 - y)] = yBytes[y * w + x]
                    }
                }
                val rotSrc = PlanarYUVLuminanceSource(rot, h, w, 0, 0, h, w, false)
                qrReader.decodeWithState(BinaryBitmap(HybridBinarizer(rotSrc))).text
            } finally {
                qrReader.reset()
            }
        }.getOrNull()
    }

    fun startRecording(
        sessionId: String,
        takeId: String,
        config: RecordingConfig,
        previewSurface: Surface?,
        outputDir: File
    ) {
        if (isRecording.get()) return
        outputDir.mkdirs()
        activeSessionId = sessionId
        activeTakeId = takeId
        activeConfig = config
        frameCounter.set(0)
        droppedFrameCounter.set(0)
        lastFrameSensorNs.set(0L)

        val videoFile = File(outputDir, "${takeId}.mp4")
        val tsFile = File(outputDir, "${takeId}_timestamps.jsonl")
        currentVideoFile = videoFile
        currentTimestampFile = tsFile
        timestampWriter = BufferedWriter(FileWriter(tsFile, false))

        setupH264Encoder(videoFile, config)
        isRecording.set(true)
        recordStartElapsedMs = SystemClock.elapsedRealtime()
        createCaptureSession(previewSurface, includeEncoder = true)
    }

    fun stopRecording(previewSurface: Surface?): CompletedRecordingResult? {
        if (!isRecording.compareAndSet(true, false)) return null
        val elapsedS = ((SystemClock.elapsedRealtime() - recordStartElapsedMs).coerceAtLeast(100L)) / 1000.0
        val frames = frameCounter.get()
        val measuredFps = frames / elapsedS

        runCatching { captureSession?.stopRepeating() }
        runCatching { mediaCodec?.signalEndOfInputStream() }
        runCatching { encoderDrainThread?.join(2500) }
        releaseEncoderAndMuxer()
        runCatching { timestampWriter?.flush(); timestampWriter?.close() }
        timestampWriter = null

        if (cameraDevice != null) {
            createCaptureSession(previewSurface, includeEncoder = false)
        }

        val vf = currentVideoFile ?: return null
        val tf = currentTimestampFile ?: return null
        if (!vf.exists() || vf.length() == 0L) return null

        val sha = computeSha256(vf)
        return CompletedRecordingResult(
            takeId = activeTakeId,
            sessionId = activeSessionId,
            videoFile = vf,
            timestampFile = tf,
            sha256Hex = sha,
            frameCount = frames,
            width = activeConfig.width,
            height = activeConfig.height,
            measuredFps = measuredFps
        )
    }

    fun close() {
        qrScanEnabled.set(false)
        if (isRecording.get()) {
            stopRecording(null)
        }
        runCatching { captureSession?.close() }
        captureSession = null
        runCatching { cameraDevice?.close() }
        cameraDevice = null
        runCatching { previewImageReader?.close() }
        previewImageReader = null
        bgThread?.quitSafely()
        bgThread = null
        bgHandler = null
    }

    private fun setupH264Encoder(outputFile: File, config: RecordingConfig) {
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, config.width, config.height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
            setInteger(MediaFormat.KEY_BIT_RATE, config.bitrateBps)
            setInteger(MediaFormat.KEY_FRAME_RATE, config.targetFps.toInt())
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        }
        val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        encoderSurface = codec.createInputSurface()
        codec.start()
        mediaCodec = codec

        mediaMuxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        muxerTrackIndex = -1
        muxerStarted = false

        encoderDrainThread = Thread({
            val info = MediaCodec.BufferInfo()
            while (true) {
                val outIndex = runCatching { codec.dequeueOutputBuffer(info, 10_000) }.getOrDefault(-1)
                when {
                    outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        if (!muxerStarted) {
                            muxerTrackIndex = mediaMuxer!!.addTrack(codec.outputFormat)
                            mediaMuxer!!.start()
                            muxerStarted = true
                        }
                    }
                    outIndex >= 0 -> {
                        val buf = codec.getOutputBuffer(outIndex)
                        if (buf != null && (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) == 0 && info.size > 0 && muxerStarted) {
                            buf.position(info.offset)
                            buf.limit(info.offset + info.size)
                            mediaMuxer!!.writeSampleData(muxerTrackIndex, buf, info)
                        }
                        codec.releaseOutputBuffer(outIndex, false)
                        if ((info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) break
                    }
                    !isRecording.get() && outIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> break
                }
            }
        }, "nuranova-h264-muxer").also { it.start() }
    }

    private fun releaseEncoderAndMuxer() {
        runCatching { mediaCodec?.stop() }
        runCatching { mediaCodec?.release() }
        mediaCodec = null
        runCatching { encoderSurface?.release() }
        encoderSurface = null
        if (muxerStarted) {
            runCatching { mediaMuxer?.stop() }
        }
        runCatching { mediaMuxer?.release() }
        mediaMuxer = null
        muxerStarted = false
    }

    @Suppress("DEPRECATION")
    private fun createCaptureSession(previewSurface: Surface?, includeEncoder: Boolean) {
        val dev = cameraDevice ?: return
        val targets = mutableListOf<Surface>()
        previewSurface?.let { if (it.isValid) targets.add(it) }
        previewImageReader?.surface?.let { if (it.isValid) targets.add(it) }
        if (includeEncoder) encoderSurface?.let { if (it.isValid) targets.add(it) }
        if (targets.isEmpty()) return

        val template = if (includeEncoder) CameraDevice.TEMPLATE_RECORD else CameraDevice.TEMPLATE_PREVIEW
        val reqBuilder = dev.createCaptureRequest(template)
        targets.forEach { reqBuilder.addTarget(it) }
        applyMocapCaptureParams(dev.id, reqBuilder, activeConfig)

        dev.createCaptureSession(targets, object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(session: CameraCaptureSession) {
                captureSession = session
                runCatching {
                    session.setRepeatingRequest(reqBuilder.build(), captureCallback, bgHandler)
                }.onFailure { onError?.invoke("Failed to start capture session: ${it.message}") }
            }

            override fun onConfigureFailed(session: CameraCaptureSession) {
                onError?.invoke("Camera2 session configuration failed for ${activeConfig.width}x${activeConfig.height} @ ${activeConfig.targetFps} FPS")
            }
        }, bgHandler)
    }

    private fun applyMocapCaptureParams(cameraId: String, req: CaptureRequest.Builder, cfg: RecordingConfig) {
        val chars = cameraManager.getCameraCharacteristics(cameraId)
        val (fpsRange, exactMatch) = chooseSupportedFpsRange(cameraId, cfg.targetFps.toInt())
        negotiatedMaxFps = fpsRange.upper.toDouble().coerceAtLeast(15.0)
        if (!exactMatch) {
            onWarning?.invoke(
                "Requested ${cfg.targetFps.toInt()} FPS is not exposed by this device's Camera2 HAL; " +
                    "using supported ${fpsRange.lower}-${fpsRange.upper} FPS."
            )
        }
        req.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fpsRange)

        if (cfg.disableStabilization) {
            req.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
            val oisModes = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION)
            if (oisModes != null && oisModes.contains(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF)) {
                req.set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF)
            }
        }
        if (cfg.lockWhiteBalance && chars.get(CameraCharacteristics.CONTROL_AWB_LOCK_AVAILABLE) == true) {
            req.set(CaptureRequest.CONTROL_AWB_LOCK, true)
        }
        if (cfg.lockFocus && cfg.focusDistanceDiopters != null) {
            req.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
            req.set(CaptureRequest.LENS_FOCUS_DISTANCE, cfg.focusDistanceDiopters)
        } else {
            req.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
        }
        if (cfg.manualExposure) {
            val aeModes = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_MODES) ?: IntArray(0)
            val expRange = chars.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
            val isoRange = chars.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
            if (aeModes.contains(CaptureRequest.CONTROL_AE_MODE_OFF) && expRange != null && isoRange != null) {
                req.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                val frameDurNs = (1_000_000_000L / fpsRange.upper.coerceAtLeast(15))
                val exp = cfg.shutterSpeedNs.coerceIn(expRange.lower, minOf(expRange.upper, frameDurNs))
                val iso = (cfg.iso ?: 800).coerceIn(isoRange.lower, isoRange.upper)
                req.set(CaptureRequest.SENSOR_EXPOSURE_TIME, exp)
                req.set(CaptureRequest.SENSOR_SENSITIVITY, iso)
                req.set(CaptureRequest.SENSOR_FRAME_DURATION, frameDurNs)
            }
        } else {
            req.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
        }
    }

    private val captureCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(
            session: CameraCaptureSession,
            request: CaptureRequest,
            result: TotalCaptureResult
        ) {
            val sensorTs = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: return
            val recvMono = SystemClock.elapsedRealtimeNanos()
            val exposure = result.get(CaptureResult.SENSOR_EXPOSURE_TIME)
            val rollingSkew = result.get(CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW)
            val frameDur = result.get(CaptureResult.SENSOR_FRAME_DURATION)
            val iso = result.get(CaptureResult.SENSOR_SENSITIVITY)

            val prev = lastFrameSensorNs.getAndSet(sensorTs)
            var instFps = negotiatedMaxFps
            if (prev > 0L && sensorTs > prev) {
                val dtNs = sensorTs - prev
                instFps = 1e9 / dtNs.toDouble()
                val expectedDtNs = (1e9 / negotiatedMaxFps.coerceAtLeast(15.0)).toLong()
                if (isRecording.get() && dtNs > expectedDtNs * 1.65) {
                    droppedFrameCounter.incrementAndGet()
                }
            }

            val totalCallbacks = previewCallbackCounter.incrementAndGet()
            if (isRecording.get()) {
                val idx = frameCounter.getAndIncrement()
                val rec = FrameTimestampRecord(
                    frameIndex = idx,
                    sensorTimestampNs = sensorTs,
                    ptsUs = sensorTs / 1000L,
                    monotonicReceiveNs = recvMono,
                    exposureNs = exposure,
                    rollingShutterSkewNs = rollingSkew,
                    frameDurationNs = frameDur
                )
                runCatching {
                    timestampWriter?.write(json.encodeToString(rec))
                    timestampWriter?.newLine()
                }
            }

            if (totalCallbacks % 10 == 0) {
                onMetricsUpdated?.invoke(instFps, droppedFrameCounter.get(), exposure, iso)
            }
        }
    }

    private fun startBackgroundThread() {
        if (bgThread != null) return
        bgThread = HandlerThread("nuranova-camera2").also { it.start() }
        bgHandler = Handler(bgThread!!.looper)
    }

    companion object {
        fun computeSha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            FileInputStream(file).use { fis ->
                val buf = ByteArray(256 * 1024)
                while (true) {
                    val read = fis.read(buf)
                    if (read <= 0) break
                    digest.update(buf, 0, read)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }

        private fun yuvImageToJpeg(image: Image, quality: Int): ByteArray? {
            return runCatching {
                val yBuf = image.planes[0].buffer
                val uBuf = image.planes[1].buffer
                val vBuf = image.planes[2].buffer
                yBuf.rewind()
                uBuf.rewind()
                vBuf.rewind()
                val ySize = yBuf.remaining()
                val uSize = uBuf.remaining()
                val vSize = vBuf.remaining()
                val nv21 = ByteArray(ySize + uSize + vSize)
                yBuf.get(nv21, 0, ySize)
                vBuf.get(nv21, ySize, vSize)
                uBuf.get(nv21, ySize + vSize, uSize)
                val out = ByteArrayOutputStream()
                YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
                    .compressToJpeg(Rect(0, 0, image.width, image.height), quality, out)
                out.toByteArray()
            }.getOrNull()
        }
    }
}
