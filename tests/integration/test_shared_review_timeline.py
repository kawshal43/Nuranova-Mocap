"""Video and character previews keep one timestamp and reopen the saved result."""

import numpy as np

from desktop.ui.app_controller import StudioController
from engine.data_format.motion_io import MotionCaptureDocument, SkeletalMotionTrack, save_motion_file
from engine.skeleton.rig_definition import BONE_NAMES, PARENT_INDICES, DEFAULT_REST_OFFSETS_M, compute_forward_kinematics_sequence
from multi_video_studio.pyav_sync import CameraSyncEstimate
from multi_video_studio.ui_page import MultiVideoCinemaLabPage
from multi_video_studio.zero_slide_physics import QualityScorecard
from tests.integration.test_multi_video_cinema_studio import _write_dummy_mp4


def test_saved_result_and_character_share_the_camera_timeline(qtbot,tmp_path):
    controller=StudioController(tmp_path/"settings.json")
    page=MultiVideoCinemaLabPage(controller,lambda _:None)
    qtbot.addWidget(page)
    videos=[_write_dummy_mp4(tmp_path/"a.mp4"),_write_dummy_mp4(tmp_path/"b.mp4")]
    try:
        page._load_source(videos)
        page.review.set_clocks({uid:CameraSyncEstimate(uid,5.,"manual",1.,"synthetic clock") for uid in page._package.motion_clips})
        count=12
        root=np.tile([0.,0.,.96],(count,1))
        root[:,0]=np.arange(count)*.01
        rotations=np.tile([0.,0.,0.,1.],(count,len(BONE_NAMES),1))
        positions,matrices=compute_forward_kinematics_sequence(root,rotations,DEFAULT_REST_OFFSETS_M)
        track=SkeletalMotionTrack(root,rotations[:,0],rotations,matrices,positions,np.ones((count,len(BONE_NAMES))))
        document=MotionCaptureDocument("timeline","timeline","timeline","timeline",30.,
            ((5+np.arange(count)/30)*1e9).astype(np.int64),BONE_NAMES,PARENT_INDICES,DEFAULT_REST_OFFSETS_M,track)
        score={name:0 for name in QualityScorecard.__dataclass_fields__}
        score.update(grade="B",calibration_mode="synthetic",notes=[],fps=30.,num_frames=count,num_cameras=2)
        document.processing_config={"scorecard":score,"review_state":page.review.state.to_dict()}
        document.sync_info=page.review.state.clocks
        result=save_motion_file(document,tmp_path/"saved.nmocap.h5")
        page.review.state.result_path=str(result)
        page.review.save()
        page._load_source(videos)
        assert page._last_result is not None
        assert page.review.document is not None
        page.review.time_slider.setValue(5200)
        assert page.slider_frame.value()==6 and page._play_frame==6
        np.testing.assert_allclose(page.viewport.canvas.skeleton_positions,positions[6])
        page.slider_frame.setValue(9)
        assert page.review.time_slider.value()==5300
        assert page.review.global_s==5.3
        page.review.time_slider.setValue(5400)
        assert page.viewport.canvas.isHidden()
        assert "No solved pose" in page.lbl_frame.text()
        page.review.time_slider.setValue(5200)
        assert not page.viewport.canvas.isHidden()
        page._toggle_play()
        assert page.review.timer.isActive() and page._playing
        page.review._toggle_play()
        assert not page.review.timer.isActive() and not page._playing
    finally:
        page.review.shutdown()
        page._play_timer.stop()
        page._gpu_timer.stop()
        controller.shutdown()
