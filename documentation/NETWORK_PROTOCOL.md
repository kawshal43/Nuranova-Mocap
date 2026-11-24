# Nuranova Mocap Studio — Network Protocol Specification (`nuranova-cam-proto/1.0`)

**Protocol Version:** `1.0.0`  
**Default Port:** `8765` (configurable)  
**Discovery Service:** mDNS / Zeroconf `_nuranova-mocap._tcp.local.`  
**Scope:** Local Area Network (LAN / Wi-Fi) only — zero internet or cloud relays required.

---

## 1. Security & Authentication

1. Studio generates a single-use, time-expiring 6-digit Pairing PIN (`CameraManager.generate_pairing_pin`).
2. Phone submits `POST /api/v1/pair` with `device_id`, `device_name`, `pin`, `role`, and dynamically probed `CameraCapabilities`.
3. Studio validates PIN in constant time (`secrets.compare_digest`), assigns a persistent `camera_uid` (`cam_01..cam_12`), and returns a 256-bit random `auth_token`.
4. All subsequent REST endpoints and WebSocket connections MUST supply `Authorization: Bearer <auth_token>` and `X-Device-Id: <device_id>`. Unauthenticated requests return `401 Unauthorized`.

---

## 2. REST Endpoints

| Method & Path | Auth | Purpose |
|---|---|---|
| `GET /api/v1/health` | Public | Minimal studio readiness probe & protocol version |
| `POST /api/v1/pair` | PIN | Pair camera device and exchange bearer token |
| `GET /api/v1/cameras` | Bearer | List connected cameras and sync status |
| `POST /api/v1/sync/clock` | Bearer | NTP-style `(t1, t2, t3)` monotonic timestamp echo |
| `POST /api/v1/sync/report` | Bearer | Submit completed `(t1, t2, t3, t4)` sample for linear drift regression |
| `POST /api/v1/telemetry` | Bearer | Report battery %, temperature °C, free storage MB, actual FPS, dropped frames |
| `POST /api/v1/transfers/init` | Bearer | Start or resume file transfer; returns `transfer_id` and `received_bytes` |
| `PUT /api/v1/transfers/{id}/chunk` | Bearer | Upload byte range at `X-Chunk-Offset` header |
| `POST /api/v1/transfers/{id}/finalize` | Bearer | Verify byte-for-byte SHA-256 checksum; sets `can_delete_local_copy` only if verified |
| `GET /api/v1/bridge/status` | Local | Blender Bridge status, active project, available takes |
| `GET /api/v1/bridge/takes/{take_id}` | Local | Fetch self-contained `.nmocap.json` payload for Blender Bridge import |

---

## 3. WebSocket Control & Low-Bandwidth Preview (`WS /ws/v1/camera/{device_id}`)

- **Phone -> Studio (Binary frames):** Low-bandwidth JPEG preview frames (`<= 320x180`, 5–10 FPS).
- **Phone -> Studio (JSON text):** `ping_sync`, `telemetry`.
- **Studio -> Phone (JSON text):** `SCHEDULE_RECORD` (carries clock-drift-compensated `scheduled_start_device_ns` + `RecordingConfig`), `STOP_RECORD`.
