package lk.nuranova.mocapcamera.ui

import android.Manifest
import android.os.Build
import android.view.SurfaceHolder
import android.view.SurfaceView
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import lk.nuranova.mocapcamera.protocol.RecordingConfig

private val TealAccent = Color(0xFF0E8A8A)
private val DarkBg = Color(0xFF0F141A)
private val PanelBg = Color(0xFF182029)
private val RecRed = Color(0xFFE53935)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MocapCameraApp(vm: MocapCameraViewModel) {
    val state by vm.uiState.collectAsState()
    val recordings by vm.localRecordings.collectAsState()
    val nav = rememberNavController()

    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = TealAccent,
            background = DarkBg,
            surface = PanelBg,
            onBackground = Color(0xFFF0F4F8),
            onSurface = Color(0xFFF0F4F8)
        )
    ) {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = { Text("Nuranova Mocap Camera", fontWeight = FontWeight.SemiBold) },
                    colors = TopAppBarDefaults.topAppBarColors(containerColor = PanelBg),
                    actions = {
                        TextButton(onClick = { nav.navigate("preview") }) { Text("Cam") }
                        TextButton(onClick = { nav.navigate("recordings") }) { Text("Takes") }
                        TextButton(onClick = { nav.navigate("transfers") }) { Text("Queue") }
                        TextButton(onClick = { nav.navigate("diagnostics") }) { Text("Diag") }
                        TextButton(onClick = { nav.navigate("settings") }) { Text("Cfg") }
                    }
                )
            }
        ) { pad ->
            NavHost(
                navController = nav,
                startDestination = if (state.config.onboardingCompleted) "preview" else "welcome",
                modifier = Modifier.padding(pad).fillMaxSize().background(DarkBg)
            ) {
                // 1. Welcome Screen
                composable("welcome") {
                    Column(
                        Modifier.fillMaxSize().padding(24.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterVertically)
                    ) {
                        Text("Nuranova Mocap Camera v1.0.0", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
                        Text("Transform this Android smartphone into a synchronized multi-camera motion capture node for Nuranova Mocap Studio.")
                        Text("All footage stays strictly on your local network. Visible REC indicators are always displayed during capture.", color = Color(0xFFA0B0C0))
                        Button(onClick = { nav.navigate("permissions") }, modifier = Modifier.fillMaxWidth()) {
                            Text("Get Started")
                        }
                        TextButton(onClick = { nav.navigate("about") }) { Text("About Nuranova Solutions") }
                    }
                }

                // 2. Permissions Screen
                composable("permissions") {
                    val perms = buildList {
                        add(Manifest.permission.CAMERA)
                        add(Manifest.permission.RECORD_AUDIO)
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                            add(Manifest.permission.POST_NOTIFICATIONS)
                        }
                    }.toTypedArray()
                    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
                        vm.probeCameraCapabilities()
                        vm.completeOnboarding()
                        nav.navigate("discovery")
                    }
                    Column(Modifier.fillMaxSize().padding(24.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                        Text("Camera & Audio Permissions", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                        Text("• Camera: Camera2 API manual exposure, focus lock, sensor timestamps, and local H.264 recording.")
                        Text("• Microphone: Audio clap/slate synchronization across multiple phones.")
                        Text("• Notifications: Persistent foreground REC indicator during capture and upload.")
                        Button(onClick = { launcher.launch(perms) }, modifier = Modifier.fillMaxWidth()) {
                            Text("Grant Permissions & Probe Camera2")
                        }
                        OutlinedButton(onClick = { nav.navigate("manual_connect") }, modifier = Modifier.fillMaxWidth()) {
                            Text("Skip to Manual Connection")
                        }
                    }
                }

                // 3. PC Discovery Screen (Android NSD)
                composable("discovery") {
                    LaunchedEffect(Unit) { vm.startDiscovery() }
                    DisposableEffect(Unit) { onDispose { vm.stopDiscovery() } }
                    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        Text("Discover Mocap Studio PC", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                        Text("Searching local network for _nuranova-mocap._tcp services...", color = Color(0xFFA0B0C0))
                        LazyColumn(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            items(state.discoveredStudios) { st ->
                                Card(colors = CardDefaults.cardColors(containerColor = PanelBg), modifier = Modifier.fillMaxWidth()) {
                                    Row(Modifier.padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
                                        Column(Modifier.weight(1f)) {
                                            Text(st.serviceName, fontWeight = FontWeight.Bold)
                                            Text("${st.host}:${st.port}", color = Color(0xFFA0B0C0))
                                        }
                                        Button(onClick = { nav.navigate("pairing/${st.host}/${st.port}") }) { Text("Pair") }
                                    }
                                }
                            }
                        }
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedButton(onClick = { nav.navigate("manual_connect") }, modifier = Modifier.weight(1f)) { Text("Manual IP") }
                            Button(onClick = { nav.navigate("preview") }, modifier = Modifier.weight(1f)) { Text("Camera View") }
                        }
                    }
                }

                // 4. Manual Connection Screen
                composable("manual_connect") {
                    var ip by remember { mutableStateOf(state.config.studioHost.ifBlank { "192.168.1.100" }) }
                    var port by remember { mutableStateOf(state.config.studioPort.toString()) }
                    Column(Modifier.fillMaxSize().padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        Text("Manual Studio Connection", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                        OutlinedTextField(value = ip, onValueChange = { ip = it }, label = { Text("Studio PC IP Address") }, modifier = Modifier.fillMaxWidth())
                        OutlinedTextField(value = port, onValueChange = { port = it }, label = { Text("Port (default 8765)") }, modifier = Modifier.fillMaxWidth())
                        Button(
                            onClick = { nav.navigate("pairing/${ip.trim()}/${port.toIntOrNull() ?: 8765}") },
                            modifier = Modifier.fillMaxWidth()
                        ) { Text("Proceed to PIN Pairing") }
                    }
                }

                // 5. Camera Pairing Screen
                composable("pairing/{host}/{port}") { backStack ->
                    val host = backStack.arguments?.getString("host") ?: ""
                    val port = backStack.arguments?.getString("port")?.toIntOrNull() ?: 8765
                    var pin by remember { mutableStateOf("") }
                    var role by remember { mutableStateOf("body") }
                    Column(Modifier.fillMaxSize().padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        Text("Pair with $host:$port", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                        Text("Enter the 6-digit Pairing PIN shown in Nuranova Mocap Studio -> Camera Manager.")
                        OutlinedTextField(value = pin, onValueChange = { pin = it }, label = { Text("6-Digit Pairing PIN") }, modifier = Modifier.fillMaxWidth())
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            listOf("body", "face", "hand_dual").forEach { r ->
                                FilterChip(selected = role == r, onClick = { role = r }, label = { Text(r) })
                            }
                        }
                        state.lastError?.let { Text(it, color = RecRed) }
                        Button(
                            onClick = {
                                vm.pairWithStudio(host, port, pin, role)
                                nav.navigate("preview")
                            },
                            enabled = pin.length >= 6,
                            modifier = Modifier.fillMaxWidth()
                        ) { Text("Pair & Connect") }
                    }
                }

                // 6 & 8. Camera Preview & Active Recording HUD Screen
                composable("preview") {
                    val borderColor = if (state.isRecording) RecRed else Color.Transparent
                    Column(Modifier.fillMaxSize().border(if (state.isRecording) 4.dp else 0.dp, borderColor)) {
                        // Top HUD Bar
                        Row(
                            Modifier.fillMaxWidth().background(PanelBg).padding(horizontal = 12.dp, vertical = 8.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column {
                                Text(
                                    "${state.config.deviceName} (${state.config.assignedCameraUid.ifBlank { "Unpaired" }})",
                                    fontWeight = FontWeight.Bold
                                )
                                Text(
                                    if (state.connected) "ONLINE • ${state.connectionMessage}" else "OFFLINE • Local Ready",
                                    color = if (state.connected) Color(0xFF4CAF50) else Color(0xFFFFB74D),
                                    style = MaterialTheme.typography.bodySmall
                                )
                            }
                            if (state.isRecording) {
                                Row(
                                    Modifier.background(RecRed, RoundedCornerShape(6.dp)).padding(horizontal = 10.dp, vertical = 4.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Box(Modifier.size(8.dp).background(Color.White, CircleShape))
                                    Spacer(Modifier.width(6.dp))
                                    Text("REC %.1fs".format(state.recordingElapsedSeconds), color = Color.White, fontWeight = FontWeight.Bold)
                                }
                            }
                        }

                        // Camera2 Live Surface Preview
                        Box(Modifier.weight(1f).fillMaxWidth().background(Color.Black)) {
                            AndroidView(
                                factory = { ctx ->
                                    SurfaceView(ctx).apply {
                                        holder.addCallback(object : SurfaceHolder.Callback {
                                            override fun surfaceCreated(h: SurfaceHolder) = vm.bindPreviewSurface(h.surface)
                                            override fun surfaceChanged(h: SurfaceHolder, f: Int, w: Int, ht: Int) {}
                                            override fun surfaceDestroyed(h: SurfaceHolder) = vm.bindPreviewSurface(null)
                                        })
                                    }
                                },
                                modifier = Modifier.fillMaxSize()
                            )
                            // Telemetry Overlay Strip
                            Column(
                                Modifier.align(Alignment.BottomStart).padding(10.dp)
                                    .background(Color(0xAA0F141A), RoundedCornerShape(8.dp)).padding(8.dp)
                            ) {
                                val shutterFrac = state.currentShutterNs?.let { if (it > 0) "1/${(1e9 / it).toInt()}s" else "Auto" } ?: "Auto"
                                val syncTxt = state.syncSummary?.let { "Sync ±%.1f ms".format(it.residualMs) } ?: "Unsynced"
                                Text("${state.recordingConfig.width}x${state.recordingConfig.height} | Act: %.1f fps | Drop: ${state.droppedFrames}".format(state.measuredFps), style = MaterialTheme.typography.bodySmall)
                                Text("Shutter: $shutterFrac | ISO: ${state.currentIso ?: "Auto"} | AF Lock: ${state.recordingConfig.lockFocus} | WB Lock: ${state.recordingConfig.lockWhiteBalance}", style = MaterialTheme.typography.bodySmall)
                                Text("$syncTxt | Batt: ${state.batteryPercent.toInt()}% | Temp: ${state.temperatureC?.let { "%.1f°C".format(it) } ?: "N/A"} | Free: ${(state.freeStorageMb / 1024.0).let { "%.1f GB".format(it) }}", style = MaterialTheme.typography.bodySmall)
                            }
                        }

                        // Bottom Action Controls
                        Row(
                            Modifier.fillMaxWidth().background(PanelBg).padding(12.dp),
                            horizontalArrangement = Arrangement.SpaceEvenly,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            OutlinedButton(onClick = { nav.navigate("rec_settings") }) { Text("Rec Cfg") }
                            Button(
                                onClick = { vm.toggleLocalRecording() },
                                colors = ButtonDefaults.buttonColors(containerColor = if (state.isRecording) RecRed else TealAccent)
                            ) {
                                Text(if (state.isRecording) "STOP REC" else "START REC", fontWeight = FontWeight.Bold)
                            }
                            OutlinedButton(onClick = { vm.runClockSync() }) { Text("Sync") }
                            OutlinedButton(onClick = { nav.navigate("discovery") }) { Text("Pair") }
                        }
                    }
                }

                // 7. Recording Settings Screen
                composable("rec_settings") {
                    var cfg by remember { mutableStateOf(state.recordingConfig) }
                    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Text("Camera Recording Settings", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                        Text("Hardware: ${state.capabilities.manufacturer} ${state.capabilities.model} (${state.capabilities.hardwareLevel})", color = Color(0xFFA0B0C0))
                        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            listOf(1280 to 720, 1920 to 1080, 2560 to 1440, 3840 to 2160).forEach { (w, h) ->
                                FilterChip(
                                    selected = cfg.height == h,
                                    onClick = { cfg = cfg.copy(width = w, height = h) },
                                    label = { Text("${h}p") }
                                )
                            }
                        }
                        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            listOf(24.0, 30.0, 60.0, 120.0).forEach { fps ->
                                FilterChip(
                                    selected = cfg.targetFps == fps,
                                    onClick = { cfg = cfg.copy(targetFps = fps) },
                                    label = { Text("${fps.toInt()}fps") }
                                )
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
                            Spacer(Modifier.width(8.dp)); Text("Disable Digital Stabilization (Recommended)")
                        }
                        Button(onClick = { vm.updateRecordingConfig(cfg); nav.popBackStack() }, modifier = Modifier.fillMaxWidth()) {
                            Text("Apply Recording Configuration")
                        }
                    }
                }

                // 9. Local Recordings Screen
                composable("recordings") {
                    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Text("Local Recordings (${recordings.size})", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            items(recordings) { rec ->
                                Card(colors = CardDefaults.cardColors(containerColor = PanelBg), modifier = Modifier.fillMaxWidth()) {
                                    Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                                        Column(Modifier.weight(1f)) {
                                            Text(rec.takeId, fontWeight = FontWeight.Bold)
                                            Text("${rec.width}x${rec.height} • ${rec.frameCount}f • %.1f fps • ${rec.transferStatus}".format(rec.measuredFps), style = MaterialTheme.typography.bodySmall)
                                            Text("SHA256: ${rec.videoSha256.take(16)}...", style = MaterialTheme.typography.bodySmall, color = Color(0xFFA0B0C0))
                                        }
                                        Button(onClick = { vm.uploadRecording(rec) }) { Text(if (rec.serverVerified) "Re-send" else "Upload") }
                                    }
                                }
                            }
                        }
                    }
                }

                // 10. Transfer Queue Screen
                composable("transfers") {
                    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Text("Resumable Transfer Queue", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            items(recordings) { rec ->
                                val progress = if (rec.videoSizeBytes > 0) (rec.bytesUploaded.toFloat() / rec.videoSizeBytes).coerceIn(0f, 1f) else 0f
                                Card(colors = CardDefaults.cardColors(containerColor = PanelBg), modifier = Modifier.fillMaxWidth()) {
                                    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                        Text("${rec.takeId} — ${rec.transferStatus.uppercase()} ${if (rec.serverVerified) "(SHA-256 VERIFIED)" else ""}", fontWeight = FontWeight.Bold)
                                        LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth())
                                    }
                                }
                            }
                        }
                    }
                }

                // 11. Diagnostics Screen
                composable("diagnostics") {
                    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("Camera2 & Sync Diagnostics", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                        Text("Device: ${state.capabilities.manufacturer} ${state.capabilities.model} (API ${state.capabilities.androidApiLevel})")
                        Text("Camera2 Hardware Level: ${state.capabilities.hardwareLevel}")
                        Text("Timestamp Source: ${state.capabilities.timestampSource}")
                        Text("Manual Exposure: ${state.capabilities.supportsManualExposure} | Manual Focus: ${state.capabilities.supportsManualFocus}")
                        Text("Sync Summary: ${state.syncSummary?.let { "Residual %.2f ms, minRTT %.2f ms (%d probes)".format(it.residualMs, it.minRttMs, it.sampleCount) } ?: "Not synced yet"}")
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(onClick = { vm.probeCameraCapabilities() }) { Text("Re-Probe Camera2") }
                            Button(onClick = { vm.reconnectStudio() }) { Text("Reconnect") }
                        }
                        HorizontalDivider()
                        Text("Recent Events", fontWeight = FontWeight.SemiBold)
                        LazyColumn(Modifier.weight(1f)) {
                            items(state.statusLog.reversed()) { line ->
                                Text(line, style = MaterialTheme.typography.bodySmall, color = Color(0xFFB0C0D0))
                            }
                        }
                    }
                }

                // 12. Application Settings Screen
                composable("settings") {
                    var name by remember { mutableStateOf(state.config.deviceName) }
                    var autoUp by remember { mutableStateOf(state.config.autoUploadAfterRecord) }
                    var retain by remember { mutableStateOf(state.config.retainLocalCopyAfterVerify) }
                    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        Text("Application Settings", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                        OutlinedTextField(value = name, onValueChange = { name = it }, label = { Text("Camera Display Name") }, modifier = Modifier.fillMaxWidth())
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Switch(checked = autoUp, onCheckedChange = { autoUp = it })
                            Spacer(Modifier.width(8.dp)); Text("Auto-transfer recordings after stop")
                        }
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Switch(checked = retain, onCheckedChange = { retain = it })
                            Spacer(Modifier.width(8.dp)); Text("Retain local backup even after SHA-256 verification")
                        }
                        Button(onClick = { vm.saveAppSettings(name, autoUp, retain) }, modifier = Modifier.fillMaxWidth()) { Text("Save Settings") }
                        OutlinedButton(onClick = { nav.navigate("about") }, modifier = Modifier.fillMaxWidth()) { Text("About Nuranova Mocap Camera") }
                    }
                }

                // 13. About Screen
                composable("about") {
                    Column(Modifier.fillMaxSize().padding(24.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Text("Nuranova Mocap Camera", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                        Text("Version 1.0.0")
                        Text("Developer: Nuranova Solutions")
                        Text("Development Team: Kawshal, Shanal, and Shehan")
                        Text("Website: https://nuranova.lk/")
                        Text("Local-first multi-camera motion capture client built with Kotlin, Jetpack Compose, Camera2 API, and MediaCodec.")
                    }
                }
            }
        }
    }
}
