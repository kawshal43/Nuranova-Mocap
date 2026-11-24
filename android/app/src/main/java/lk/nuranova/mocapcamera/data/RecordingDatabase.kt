package lk.nuranova.mocapcamera.data

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import java.util.UUID

@Entity(tableName = "local_recordings")
data class LocalRecordingEntity(
    @PrimaryKey val takeId: String,
    val sessionId: String,
    val videoPath: String,
    val timestampPath: String,
    val videoSizeBytes: Long,
    val videoSha256: String,
    val frameCount: Int,
    val width: Int,
    val height: Int,
    val measuredFps: Double,
    val transferStatus: String, // "pending", "uploading", "verified", "failed", "recovered"
    val bytesUploaded: Long = 0L,
    val serverVerified: Boolean = false,
    val createdAtMs: Long = System.currentTimeMillis()
)

@Dao
interface LocalRecordingDao {
    @Query("SELECT * FROM local_recordings ORDER BY createdAtMs DESC")
    fun observeAll(): Flow<List<LocalRecordingEntity>>

    @Query("SELECT * FROM local_recordings WHERE serverVerified = 0 ORDER BY createdAtMs ASC")
    suspend fun getPendingUploads(): List<LocalRecordingEntity>

    @Query("SELECT * FROM local_recordings WHERE takeId = :takeId LIMIT 1")
    suspend fun getById(takeId: String): LocalRecordingEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(entity: LocalRecordingEntity)

    @Query("UPDATE local_recordings SET transferStatus = :status, bytesUploaded = :bytes, serverVerified = :verified WHERE takeId = :takeId")
    suspend fun updateTransfer(takeId: String, status: String, bytes: Long, verified: Boolean)

    @Query("DELETE FROM local_recordings WHERE takeId = :takeId")
    suspend fun deleteById(takeId: String)
}

@Database(entities = [LocalRecordingEntity::class], version = 1, exportSchema = false)
abstract class MocapDatabase : RoomDatabase() {
    abstract fun recordingDao(): LocalRecordingDao

    companion object {
        @Volatile
        private var INSTANCE: MocapDatabase? = null

        fun getInstance(context: Context): MocapDatabase =
            INSTANCE ?: synchronized(this) {
                INSTANCE ?: Room.databaseBuilder(
                    context.applicationContext,
                    MocapDatabase::class.java,
                    "nuranova_mocap_camera.db"
                ).build().also { INSTANCE = it }
            }
    }
}

private val Context.dataStore by preferencesDataStore(name = "nuranova_camera_prefs")

data class StoredAppConfig(
    val deviceId: String = "",
    val deviceName: String = android.os.Build.MODEL ?: "Android Camera",
    val studioHost: String = "",
    val studioPort: Int = 8765,
    val authToken: String = "",
    val assignedCameraUid: String = "",
    val cameraRole: String = "body",
    val autoUploadAfterRecord: Boolean = true,
    val retainLocalCopyAfterVerify: Boolean = true,
    val onboardingCompleted: Boolean = false
)

class SettingsStore(private val context: Context) {
    private object Keys {
        val DEVICE_ID = stringPreferencesKey("device_id")
        val DEVICE_NAME = stringPreferencesKey("device_name")
        val STUDIO_HOST = stringPreferencesKey("studio_host")
        val STUDIO_PORT = intPreferencesKey("studio_port")
        val AUTH_TOKEN = stringPreferencesKey("auth_token")
        val CAMERA_UID = stringPreferencesKey("camera_uid")
        val CAMERA_ROLE = stringPreferencesKey("camera_role")
        val AUTO_UPLOAD = booleanPreferencesKey("auto_upload")
        val RETAIN_LOCAL = booleanPreferencesKey("retain_local")
        val ONBOARDING_DONE = booleanPreferencesKey("onboarding_done")
    }

    val configFlow: Flow<StoredAppConfig> = context.dataStore.data.map { p ->
        StoredAppConfig(
            deviceId = p[Keys.DEVICE_ID] ?: "",
            deviceName = p[Keys.DEVICE_NAME] ?: (android.os.Build.MODEL ?: "Android Camera"),
            studioHost = p[Keys.STUDIO_HOST] ?: "",
            studioPort = p[Keys.STUDIO_PORT] ?: 8765,
            authToken = p[Keys.AUTH_TOKEN] ?: "",
            assignedCameraUid = p[Keys.CAMERA_UID] ?: "",
            cameraRole = p[Keys.CAMERA_ROLE] ?: "body",
            autoUploadAfterRecord = p[Keys.AUTO_UPLOAD] ?: true,
            retainLocalCopyAfterVerify = p[Keys.RETAIN_LOCAL] ?: true,
            onboardingCompleted = p[Keys.ONBOARDING_DONE] ?: false
        )
    }

    suspend fun ensureDeviceId(): String {
        var id = ""
        context.dataStore.edit { p ->
            val existing = p[Keys.DEVICE_ID]
            if (existing.isNullOrBlank()) {
                id = "android_${UUID.randomUUID().toString().replace("-", "").take(12)}"
                p[Keys.DEVICE_ID] = id
            } else {
                id = existing
            }
        }
        return id
    }

    suspend fun savePairing(host: String, port: Int, token: String, cameraUid: String, role: String) {
        context.dataStore.edit { p ->
            p[Keys.STUDIO_HOST] = host
            p[Keys.STUDIO_PORT] = port
            p[Keys.AUTH_TOKEN] = token
            p[Keys.CAMERA_UID] = cameraUid
            p[Keys.CAMERA_ROLE] = role
        }
    }

    suspend fun updateSettings(deviceName: String, autoUpload: Boolean, retainLocal: Boolean) {
        context.dataStore.edit { p ->
            p[Keys.DEVICE_NAME] = deviceName
            p[Keys.AUTO_UPLOAD] = autoUpload
            p[Keys.RETAIN_LOCAL] = retainLocal
        }
    }

    suspend fun markOnboardingComplete() {
        context.dataStore.edit { p -> p[Keys.ONBOARDING_DONE] = true }
    }
}
