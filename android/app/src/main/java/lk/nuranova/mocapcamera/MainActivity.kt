package lk.nuranova.mocapcamera

import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import lk.nuranova.mocapcamera.ui.MocapCameraApp
import lk.nuranova.mocapcamera.ui.MocapCameraViewModel

class MainActivity : ComponentActivity() {
    private val viewModel: MocapCameraViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContent {
            MocapCameraApp(viewModel)
        }
    }
}
