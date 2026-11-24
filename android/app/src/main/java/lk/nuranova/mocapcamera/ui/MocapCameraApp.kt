package lk.nuranova.mocapcamera.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.SurfaceTexture
import android.os.Build
import android.view.Surface
import android.view.TextureView
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.QrCodeScanner
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Sync
import androidx.compose.material.icons.filled.VideoLibrary
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material.icons.filled.Wifi
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat

private val TealAccent = Color(0xFF14A3A3)
private val DarkBg = Color(0xFF0D131A)
private val PanelBg = Color(0xFF16202B)
private val CardBg = Color(0xFF1C2735)
private val OkGreen = Color(0xFF22C55E)
private val WarnAmber = Color(0xFFF59E0B)
private val RecRed = Color(0xFFE53935)
private val MutedText = Color(0xFF94A6B8)

private enum class BottomTab(val label: String) {
    CAMERA("Camera"),
    CONNECT("Connect"),
    TAKES("Takes"),
    SETTINGS("Settings")
}

@Composable
fun MocapCameraApp(vm: MocapCameraViewModel) {
    val state by vm.uiState.collectAsState()
    val recordings by vm.localRecordings.collectAsState()
    val ctx = LocalContext.current

    var hasCameraPerm by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(ctx, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }
    var activeTab by remember {
        mutableStateOf(if (state.config.authToken.isNotBlank()) BottomTab.CAMERA else BottomTab.CONNECT)
    }

    // Automatically switch to Camera tab once QR/PIN pairing succeeds
    LaunchedEffect(state.pairingState) {
        if (state.pairingState is PairingState.Paired) {
            activeTab = BottomTab.CAMERA
            vm.resetPairingState()
        }
    }

    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = TealAccent,
            onPrimary = Color.White,
            secondary = TealAccent,
            onSecondary = Color.White,
            background = DarkBg,
            surface = PanelBg,
            onBackground = Color(0xFFF0F4F8),
            onSurface = Color(0xFFF0F4F8)
        )
    ) {
        if (!hasCameraPerm) {
            PermissionOnboardingScreen(
                onGranted = {
                    hasCameraPerm = true
                    vm.probeCameraCapabilities()
                    vm.completeOnboarding()
                }
            )
            return@MaterialTheme
        }

        // Full-screen QR Scanner overlay (reuses Camera2 cleanly without conflicts!)
        if (state.isQrScanning) {
            QrScannerOverlay(
                vm = vm,
                onClose = { vm.setQrScanning(false) }
            )
            return@MaterialTheme
        }

        Scaffold(
            containerColor = DarkBg,
            bottomBar = {
                if (!state.isRecording) {
                    NavigationBar(containerColor = PanelBg, tonalElevation = 4.dp) {
                        NavigationBarItem(
                            selected = activeTab == BottomTab.CAMERA,
                            onClick = { activeTab = BottomTab.CAMERA },
                            icon = { Icon(Icons.Default.Videocam, contentDescription = "Camera") },
                            label = { Text("Camera") }
                        )
                        NavigationBarItem(
                            selected = activeTab == BottomTab.CONNECT,
                            onClick = { activeTab = BottomTab.CONNECT },
                            icon = {
                                BadgedBox(
                                    badge = {
                                        if (!state.connected) {
                                            Badge(containerColor = WarnAmber)
                                        } else {
                                            Badge(containerColor = OkGreen)
                                        }
                                    }
                                ) {
                                    Icon(Icons.Default.Wifi, contentDescription = "Connect")
                                }
                            },
                            label = { Text("Connect") }
                        )
                        NavigationBarItem(
                            selected = activeTab == BottomTab.TAKES,
                            onClick = { activeTab = BottomTab.TAKES },
                            icon = {
                                BadgedBox(
                                    badge = {
                                        val pending = recordings.count { !it.serverVerified }
                                        if (pending > 0) Badge { Text("$pending") }
                                    }
                                ) {
                                    Icon(Icons.Default.VideoLibrary, contentDescription = "Takes")
                                }
                            },
                            label = { Text("Takes") }
                        )
                        NavigationBarItem(
                            selected = activeTab == BottomTab.SETTINGS,
                            onClick = { activeTab = BottomTab.SETTINGS },
                            icon = { Icon(Icons.Default.Settings, contentDescription = "Settings") },
                            label = { Text("Settings") }
                        )
                    }
                }
            }
        ) { pad ->
            Box(Modifier.padding(pad).fillMaxSize()) {
                when (activeTab) {
                    BottomTab.CAMERA -> CameraScreen(
                        vm = vm,
                        state = state,
                        onOpenConnect = { activeTab = BottomTab.CONNECT }
                    )
                    BottomTab.CONNECT -> ConnectHubScreen(
                        vm = vm,
                        state = state
                    )
                    BottomTab.TAKES -> TakesScreen(
                        vm = vm,
                        recordings = recordings
                    )
                    BottomTab.SETTINGS -> SettingsScreen(
                        vm = vm,
                        state = state
                    )
                }
            }
        }
    }
}

@Composable
private fun PermissionOnboardingScreen(onGranted: () -> Unit) {
    val perms = buildList {
        add(Manifest.permission.CAMERA)
        add(Manifest.permission.RECORD_AUDIO)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            add(Manifest.permission.POST_NOTIFICATIONS)
        }
    }.toTypedArray()

    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { res ->
        if (res[Manifest.permission.CAMERA] == true) {
            onGranted()
        }
    }

    Column(
        Modifier.fillMaxSize().background(DarkBg).padding(28.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp, Alignment.CenterVertically),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Icon(Icons.Default.Videocam, contentDescription = null, tint = TealAccent, modifier = Modifier.size(64.dp))
        Text("Nuranova Mocap Camera", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
        Text(
            "Turn this smartphone into a synchronized motion capture camera on your local Wi-Fi network.",
            color = MutedText
        )
        Card(colors = CardDefaults.cardColors(containerColor = CardBg), modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Permissions Needed:", fontWeight = FontWeight.Bold)
                Text("• Camera — Camera2 manual exposure, sensor timestamps & QR pairing", color = MutedText)
                Text("• Microphone — Audio slate clap synchronization", color = MutedText)
                Text("• Notifications — Visible REC indicator during capture", color = MutedText)
            }
        }
        Button(
            onClick = { launcher.launch(perms) },
            modifier = Modifier.fillMaxWidth().height(52.dp),
            shape = RoundedCornerShape(12.dp)
        ) {
            Text("Grant Permissions & Continue", fontWeight = FontWeight.Bold)
        }
    }
}

@Composable
private fun QrScannerOverlay(vm: MocapCameraViewModel, onClose: () -> Unit) {
    val state by vm.uiState.collectAsState()
    LaunchedEffect(Unit) { vm.setQrScanning(true) }
    DisposableEffect(Unit) { onDispose { vm.cameraController.qrScanEnabled.set(false) } }

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        CameraTextureView(vm = vm, state = state, modifier = Modifier.fillMaxSize())

        // Targeting frame overlay
        Box(
            Modifier
                .size(250.dp)
                .align(Alignment.Center)
                .border(3.dp, TealAccent, RoundedCornerShape(18.dp))
        )

        Column(
            Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth()
                .background(Color(0xCC0D131A))
                .statusBarsPadding()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                Text("Scan Studio QR Code", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, color = Color.White)
                IconButton(onClick = onClose) {
                    Icon(Icons.Default.Close, contentDescription = "Close", tint = Color.White)
                }
            }
            Text(
                "Point your phone at the QR code in Mocap Studio → 2. Cameras & Pairing",
                style = MaterialTheme.typography.bodySmall,
                color = MutedText
            )
        }
    }
}

@Composable
private fun CameraTextureView(vm: MocapCameraViewModel, state: MocapUiState, modifier: Modifier = Modifier) {
    AndroidView(
        factory = { ctx ->
            val previewSize = runCatching {
                vm.cameraController.chooseSupportedPreviewSize(
                    state.recordingConfig.cameraLens,
                    state.recordingConfig.width,
                    state.recordingConfig.height
                )
            }.getOrDefault(android.util.Size(1280, 720))
            TextureView(ctx).apply {
                surfaceTextureListener = object : TextureView.SurfaceTextureListener {
                    private var boundSurface: Surface? = null
                    override fun onSurfaceTextureAvailable(st: SurfaceTexture, w: Int, h: Int) {
                        st.setDefaultBufferSize(previewSize.width, previewSize.height)
                        boundSurface = Surface(st)
                        vm.bindPreviewSurface(boundSurface)
                    }
                    override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, w: Int, h: Int) {}
                    override fun onSurfaceTextureDestroyed(st: SurfaceTexture): Boolean {
                        vm.bindPreviewSurface(null)
                        boundSurface?.release()
                        boundSurface = null
                        return true
                    }
                    override fun onSurfaceTextureUpdated(st: SurfaceTexture) {}
                }
            }
        },
        modifier = modifier
    )
}

@Composable
private fun CameraScreen(
    vm: MocapCameraViewModel,
    state: MocapUiState,
    onOpenConnect: () -> Unit
) {
    val borderColor = if (state.isRecording) RecRed else Color.Transparent
    Box(Modifier.fillMaxSize().border(if (state.isRecording) 4.dp else 0.dp, borderColor).background(Color.Black)) {
        CameraTextureView(vm = vm, state = state, modifier = Modifier.fillMaxSize())

        // Top HUD Status Pill + Error Banner
        Column(
            Modifier.align(Alignment.TopCenter).fillMaxWidth().padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                val pillBg = if (state.connected) Color(0xDD113826) else Color(0xDD3B2410)
                val dotCol = if (state.connected) OkGreen else WarnAmber
                val syncTxt = state.syncSummary?.let { " · ±%.1fms".format(it.residualMs) } ?: ""
                val statusLabel = if (state.connected) {
                    "ONLINE · ${state.config.assignedCameraUid.ifBlank { "Paired" }}$syncTxt"
                } else if (state.needsRePair) {
                    "RE-PAIR NEEDED · Tap here"
                } else {
                    "OFFLINE · Tap to Connect"
                }
                Row(
                    Modifier
                        .clip(RoundedCornerShape(20.dp))
                        .background(pillBg)
                        .clickable { onOpenConnect() }
                        .padding(horizontal = 12.dp, vertical = 7.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(7.dp)
                ) {
                    Box(Modifier.size(9.dp).background(dotCol, CircleShape))
                    Text(statusLabel, color = Color.White, style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold)
                }

                if (state.isRecording) {
                    Row(
                        Modifier.background(RecRed, RoundedCornerShape(20.dp)).padding(horizontal = 12.dp, vertical = 7.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Box(Modifier.size(8.dp).background(Color.White, CircleShape))
                        Spacer(Modifier.width(6.dp))
                        Text("REC %.1fs".format(state.recordingElapsedSeconds), color = Color.White, fontWeight = FontWeight.Bold)
                    }
                }
            }

            if (state.lastError != null || state.needsRePair) {
                Card(
                    colors = CardDefaults.cardColors(containerColor = Color(0xEE5C1B1B)),
                    shape = RoundedCornerShape(10.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Row(
                        Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(Icons.Default.Warning, contentDescription = null, tint = Color(0xFFFCA5A5), modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(8.dp))
                        Text(
                            state.lastError ?: "Desktop Studio restarted or unpaired. Open Connect to re-pair.",
                            color = Color.White,
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.weight(1f)
                        )
                        TextButton(onClick = { vm.dismissError(); onOpenConnect() }) {
                            Text("Fix", color = Color(0xFFFDE68A), fontWeight = FontWeight.Bold)
                        }
                    }
                }
            }
        }

        // Bottom Telemetry + Thumb-Friendly Record Bar
        Column(
            Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .background(Color(0xCC0D131A))
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            val shutterFrac = state.currentShutterNs?.let { if (it > 0) "1/${(1e9 / it).toInt()}s" else "Auto" } ?: "Auto"
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(
                    "${state.recordingConfig.height}p · %.1f fps · Drop ${state.droppedFrames}".format(state.measuredFps),
                    style = MaterialTheme.typography.bodySmall,
                    color = Color.White
                )
                Text(
                    "Shut $shutterFrac · Batt ${state.batteryPercent.toInt()}% · ${"%.1f".format(state.freeStorageMb / 1024.0)}GB",
                    style = MaterialTheme.typography.bodySmall,
                    color = MutedText
                )
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly,
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedButton(
                    onClick = { vm.runClockSync() },
                    enabled = state.connected
                ) {
                    Icon(Icons.Default.Sync, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(4.dp))
                    Text("Sync")
                }

                Button(
                    onClick = { vm.toggleLocalRecording() },
                    colors = ButtonDefaults.buttonColors(containerColor = if (state.isRecording) RecRed else TealAccent),
                    shape = RoundedCornerShape(28.dp),
                    modifier = Modifier.height(52.dp).widthIn(min = 156.dp)
                ) {
                    Text(
                        if (state.isRecording) "■ STOP REC" else "● RECORD",
                        fontWeight = FontWeight.ExtraBold,
                        style = MaterialTheme.typography.titleSmall
                    )
                }

                OutlinedButton(onClick = { vm.setQrScanning(true) }) {
                    Icon(Icons.Default.QrCodeScanner, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(4.dp))
                    Text("QR")
                }
            }
        }
    }
}

@Composable
private fun ConnectHubScreen(vm: MocapCameraViewModel, state: MocapUiState) {
    LaunchedEffect(Unit) { vm.startDiscovery() }
    DisposableEffect(Unit) { onDispose { vm.stopDiscovery() } }

    var ip by remember(state.config.studioHost) { mutableStateOf(state.config.studioHost) }
    var port by remember(state.config.studioPort) { mutableStateOf(state.config.studioPort.toString()) }
    var pin by remember { mutableStateOf("") }
    var role by remember(state.config.cameraRole) { mutableStateOf(state.config.cameraRole.ifBlank { "body" }) }

    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Text("Connect to Studio PC", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)

        // Live connection status banner
        Card(
            colors = CardDefaults.cardColors(
                containerColor = if (state.connected) Color(0xFF133325) else CardBg
            ),
            modifier = Modifier.fillMaxWidth()
        ) {
            Row(Modifier.padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    if (state.connected) Icons.Default.CheckCircle else Icons.Default.Wifi,
                    contentDescription = null,
                    tint = if (state.connected) OkGreen else WarnAmber
                )
                Spacer(Modifier.width(10.dp))
                Column(Modifier.weight(1f)) {
                    Text(
                        if (state.connected) "Connected (${state.config.assignedCameraUid})" else "Not Connected",
                        fontWeight = FontWeight.Bold
                    )
                    Text(state.connectionMessage, style = MaterialTheme.typography.bodySmall, color = MutedText)
                }
                if (!state.connected && state.config.studioHost.isNotBlank() && state.config.authToken.isNotBlank() && !state.needsRePair) {
                    TextButton(onClick = { vm.reconnectStudio() }) { Text("Retry") }
                }
            }
        }

        // 1. Primary 1-Tap QR Code Scan CTA
        Button(
            onClick = { vm.setQrScanning(true) },
            modifier = Modifier.fillMaxWidth().height(54.dp),
            shape = RoundedCornerShape(12.dp)
        ) {
            Icon(Icons.Default.QrCodeScanner, contentDescription = null)
            Spacer(Modifier.width(10.dp))
            Text("Scan QR Code in Mocap Studio (Fastest)", fontWeight = FontWeight.Bold)
        }

        // 2. Auto-Discovered Studio PCs (mDNS + UDP Beacon)
        Card(colors = CardDefaults.cardColors(containerColor = CardBg), modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                    Text("Studio PCs on Wi-Fi", fontWeight = FontWeight.Bold)
                    IconButton(onClick = { vm.startDiscovery() }, modifier = Modifier.size(32.dp)) {
                        Icon(Icons.Default.Refresh, contentDescription = "Refresh", tint = TealAccent)
                    }
                }
                if (state.discoveredStudios.isEmpty()) {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                        Text(
                            "Searching local network (mDNS + UDP :8766)... Ensure phone & PC share the same Wi-Fi.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MutedText
                        )
                    }
                } else {
                    state.discoveredStudios.forEach { st ->
                        Row(
                            Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(8.dp))
                                .background(PanelBg)
                                .clickable {
                                    ip = st.host
                                    port = st.port.toString()
                                    vm.testStudioConnection(st.host, st.port)
                                }
                                .padding(12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(Modifier.weight(1f)) {
                                Text(st.serviceName, fontWeight = FontWeight.SemiBold)
                                Text("${st.host}:${st.port}  (${st.source})", style = MaterialTheme.typography.bodySmall, color = MutedText)
                            }
                            Text("Select", color = TealAccent, fontWeight = FontWeight.Bold)
                        }
                    }
                }
            }
        }

        // 3. Manual IP + Test Connection + PIN Pairing
        Card(colors = CardDefaults.cardColors(containerColor = CardBg), modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("Pair with IP & 6-Digit PIN", fontWeight = FontWeight.Bold)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(
                        value = ip,
                        onValueChange = { ip = it },
                        label = { Text("PC IP (e.g. 192.168.8.124)") },
                        singleLine = true,
                        modifier = Modifier.weight(2f)
                    )
                    OutlinedTextField(
                        value = port,
                        onValueChange = { port = it },
                        label = { Text("Port") },
                        singleLine = true,
                        modifier = Modifier.weight(1f)
                    )
                }

                OutlinedTextField(
                    value = pin,
                    onValueChange = { pin = it.filter { ch -> ch.isDigit() }.take(6) },
                    label = { Text("6-Digit PIN from Mocap Studio") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf("body" to "Body", "face" to "Face", "hand_dual" to "Hands").forEach { (key, lbl) ->
                        FilterChip(selected = role == key, onClick = { role = key }, label = { Text(lbl) })
                    }
                }

                // Inline Pairing / Test Status Feedback (NEVER hidden!)
                when (val ps = state.pairingState) {
                    is PairingState.Testing -> Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
                        Text("Testing connection to $ip:$port...", style = MaterialTheme.typography.bodySmall)
                    }
                    is PairingState.Pairing -> Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
                        Text("Pairing with Mocap Studio...", style = MaterialTheme.typography.bodySmall)
                    }
                    is PairingState.Reachable -> Text(
                        "✓ ${ps.message}",
                        color = OkGreen,
                        style = MaterialTheme.typography.bodySmall,
                        fontWeight = FontWeight.SemiBold
                    )
                    is PairingState.Error -> Card(
                        colors = CardDefaults.cardColors(containerColor = Color(0xFF47181B)),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(ps.message, color = Color(0xFFFCA5A5), style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(10.dp))
                    }
                    else -> {}
                }

                Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                    OutlinedButton(
                        onClick = { vm.testStudioConnection(ip, port.toIntOrNull() ?: 8765) },
                        enabled = state.pairingState !is PairingState.Testing && state.pairingState !is PairingState.Pairing,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Test PC")
                    }
                    Button(
                        onClick = { vm.pairWithStudio(ip, port.toIntOrNull() ?: 8765, pin, role) },
                        enabled = ip.isNotBlank() && pin.length >= 6 && state.pairingState !is PairingState.Pairing,
                        modifier = Modifier.weight(1.4f)
                    ) {
                        Text("Pair & Connect", fontWeight = FontWeight.Bold)
                    }
                }
            }
        }
    }
}

@Composable
private fun TakesScreen(
    vm: MocapCameraViewModel,
    recordings: List<lk.nuranova.mocapcamera.data.LocalRecordingEntity>
) {
    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text("Recorded Takes & Transfers (${recordings.size})", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        if (recordings.isEmpty()) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("No takes recorded yet. Press RECORD on the Camera tab or start from Desktop Studio.", color = MutedText)
            }
        } else {
            LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                items(recordings) { rec ->
                    val progress = if (rec.videoSizeBytes > 0) (rec.bytesUploaded.toFloat() / rec.videoSizeBytes).coerceIn(0f, 1f) else 0f
                    Card(colors = CardDefaults.cardColors(containerColor = CardBg), modifier = Modifier.fillMaxWidth()) {
                        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Column(Modifier.weight(1f)) {
                                    Text(rec.takeId, fontWeight = FontWeight.Bold)
                                    Text(
                                        "${rec.width}x${rec.height} · ${rec.frameCount}f · %.1f fps".format(rec.measuredFps),
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MutedText
                                    )
                                }
                                Button(onClick = { vm.uploadRecording(rec) }) {
                                    Text(if (rec.serverVerified) "Re-send" else "Upload")
                                }
                            }
                            LinearProgressIndicator(progress = { if (rec.serverVerified) 1f else progress }, modifier = Modifier.fillMaxWidth())
                            Text(
                                if (rec.serverVerified) "✓ Verified SHA-256 on Studio PC" else "Status: ${rec.transferStatus.uppercase()}",
                                style = MaterialTheme.typography.labelSmall,
                                color = if (rec.serverVerified) OkGreen else WarnAmber
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun SettingsScreen(vm: MocapCameraViewModel, state: MocapUiState) {
    var cfg by remember(state.recordingConfig) { mutableStateOf(state.recordingConfig) }
    var name by remember(state.config.deviceName) { mutableStateOf(state.config.deviceName) }
    var autoUp by remember(state.config.autoUploadAfterRecord) { mutableStateOf(state.config.autoUploadAfterRecord) }
    var retain by remember(state.config.retainLocalCopyAfterVerify) { mutableStateOf(state.config.retainLocalCopyAfterVerify) }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Text("Camera & App Settings", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)

        // Recording Config Card
        Card(colors = CardDefaults.cardColors(containerColor = CardBg), modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Video Capture Settings", fontWeight = FontWeight.Bold)
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    listOf(1280 to 720, 1920 to 1080, 2560 to 1440, 3840 to 2160).forEach { (w, h) ->
                        FilterChip(selected = cfg.height == h, onClick = { cfg = cfg.copy(width = w, height = h) }, label = { Text("${h}p") })
                    }
                }
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    listOf(24.0, 30.0, 60.0, 120.0).forEach { fps ->
                        FilterChip(selected = cfg.targetFps == fps, onClick = { cfg = cfg.copy(targetFps = fps) }, label = { Text("${fps.toInt()}fps") })
                    }
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Switch(checked = cfg.manualExposure, onCheckedChange = { cfg = cfg.copy(manualExposure = it) })
                    Spacer(Modifier.width(8.dp)); Text("Manual Exposure (1/250s target)")
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Switch(checked = cfg.lockFocus, onCheckedChange = { cfg = cfg.copy(lockFocus = it) })
                    Spacer(Modifier.width(8.dp)); Text("Lock Focus")
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Switch(checked = cfg.disableStabilization, onCheckedChange = { cfg = cfg.copy(disableStabilization = it) })
                    Spacer(Modifier.width(8.dp)); Text("Disable Digital Stabilization")
                }
                Button(onClick = { vm.updateRecordingConfig(cfg) }, modifier = Modifier.fillMaxWidth()) {
                    Text("Apply Camera Config")
                }
            }
        }

        // Device & Transfer Settings
        Card(colors = CardDefaults.cardColors(containerColor = CardBg), modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Device & Transfer", fontWeight = FontWeight.Bold)
                OutlinedTextField(value = name, onValueChange = { name = it }, label = { Text("Camera Display Name") }, modifier = Modifier.fillMaxWidth())
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Switch(checked = autoUp, onCheckedChange = { autoUp = it })
                    Spacer(Modifier.width(8.dp)); Text("Auto-upload takes after stop")
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Switch(checked = retain, onCheckedChange = { retain = it })
                    Spacer(Modifier.width(8.dp)); Text("Keep local backup after SHA-256 verify")
                }
                Button(onClick = { vm.saveAppSettings(name, autoUp, retain) }, modifier = Modifier.fillMaxWidth()) {
                    Text("Save App Settings")
                }
            }
        }

        // Diagnostics & Log Card
        Card(colors = CardDefaults.cardColors(containerColor = CardBg), modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("Diagnostics & Recent Logs", fontWeight = FontWeight.Bold)
                Text("${state.capabilities.manufacturer} ${state.capabilities.model} · API ${state.capabilities.androidApiLevel}", style = MaterialTheme.typography.bodySmall, color = MutedText)
                Text("Camera2: ${state.capabilities.hardwareLevel} · Timestamp: ${state.capabilities.timestampSource}", style = MaterialTheme.typography.bodySmall, color = MutedText)
                Text(
                    "Clock Sync: ${state.syncSummary?.let { "±%.2f ms (minRTT %.1f ms)".format(it.residualMs, it.minRttMs) } ?: "Not synced"}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MutedText
                )
                OutlinedButton(onClick = { vm.probeCameraCapabilities() }) { Text("Re-Probe Camera2") }
                HorizontalDivider()
                state.statusLog.takeLast(8).reversed().forEach { line ->
                    Text(line, style = MaterialTheme.typography.labelSmall, color = MutedText)
                }
            }
        }
    }
}
