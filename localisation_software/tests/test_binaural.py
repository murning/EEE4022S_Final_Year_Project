from binaural import Binaural
import numpy as np

b = Binaural(room_dim=np.r_[3., 3., 2.5],
             max_order=17,
             speed_of_sound=343,
             inter_aural_distance=0.2,
             mic_height=1)


def test_source_location_90():
    location = b.source_location(0.5, 90)

    assert location[0] == 1.5
    assert location[1] == 2
    assert location[2] == 1.25


def test_source_location_180():
    location = b.source_location(1, 180)

    assert location[0] == 0.5
    assert location[1] == 1.5
    assert location[2] == 1.25


def test_source_location_225():
    location = b.source_location(1.2, 225)

    assert location[0] == 0.6515
    assert location[1] == 0.6515
    assert location[2] == 1.25

def test_mic_position_0():
    position = b.mic_position(np.array([1.5,1.5]), 0)

    #left
    assert position[0][0] == 1.4
    assert position[1][0] == 1.5
    assert position[2][0] == 1

    #right
    assert position[0][1] == 1.6
    assert position[1][1] == 1.5
    assert position[2][1] == 1


def test_mic_position_225_shift():
    position = b.mic_position(np.array([1,1]), 225)

    #left
    assert position[0][0] == 1.0707
    assert position[1][0] == 1.0707
    assert position[2][0] == 1

    #right
    assert position[0][1] == 0.9293
    assert position[1][1] == 0.9293
    assert position[2][1] == 1



def test_true_azimuth_90():

    azimuth = b.true_azimuth(np.array([1.5,1.5]),0,np.array([1.5,3]))

    assert azimuth == 90.0


def test_true_azimuth_0():

    azimuth = b.true_azimuth(np.array([1.5,1.5]),0,np.array([3,1.5]))

    assert azimuth == 0.0


def test_true_azimuth_180():

    azimuth = b.true_azimuth(np.array([1.5,1.5]),0,np.array([1,1.5]))

    assert azimuth == 180.0


def test_true_azimuth_270():

    azimuth = b.true_azimuth(np.array([1.5,1.5]),0,np.array([1.5,0]))

    assert azimuth == 270.0


def test_true_azimuth_shift_rot_15():

    azimuth = b.true_azimuth(np.array([1,2]),15,np.array([2.3192,2.0736]))

    assert azimuth == 348


def test_true_azimuth_shift_rot_345():
    azimuth = b.true_azimuth(np.array([1, 2]), 345, np.array([2.3192, 2.0736]))

    assert azimuth == 18
