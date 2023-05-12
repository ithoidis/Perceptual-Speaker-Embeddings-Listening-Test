"""
@Author: Iordanis Thoidis
@Date: 11/5/23
@Link: https://github.com/ithoidis/Perceptual-Speaker-Embeddings-Listening-Test
"""

import time
import os
from psychopy import prefs, core
prefs.hardware['audioLib'] = ['PTB']
from psychopy.sound.backend_ptb import SoundPTB
import numpy as np
import pandas as pd
from psychopy import visual, event
from utils import rms, apply_raised_cosine_ramp, silence, attenuate_sound, read_audio
from export_stimuli import check_data


def speaker_discrimination_block(participant, session, ear='both', level=-5, n_trials=40, noise=None, snr=None,
                                 feedback=False, reverse_order=False):
    """
    :param participant: (string) The ID of the participant or 'training'. It is used as an identifier for the results
    file.
    :param session: Number of Session to run. In [1], 2 sessions of 40 trials were used.
    :param ear: Both for diotic stimulus, left/right for lateral.
    :param level: Sound presentation level in dB FS. Make sure to calibrate the headphone output first.
    :param n_trials: Number of trials to run
    :param noise: None for no noise, 'iltass' for speech-shaped noise, and 'babble' for babble noise. Default=None
    :param snr: The signal-to-noise ratio. Default=None
    :param feedback: Whether to provide feedback to the participant
    :param reverse_order: If True, the stimulus order presentation is reversed (speaker1 - silence - speaker2).
    Default=False

    [1] Thoidis, Iordanis, Clément Gaultier, and Tobias Goehring. "Perceptual Analysis of Speaker Embeddings for Voice
    Discrimination between Machine And Human Listening." In ICASSP 2023-2023 IEEE International Conference on Acoustics,
    Speech and Signal Processing (ICASSP), pp. 1-5. IEEE, 2023.
    """
    # Presentation level in dB FS
    assert level <= 0.0
    assert ear in ['left', 'right', 'both']
    assert noise in [None, 'babble', 'iltass']
    assert snr in [None, 2, 5, 6]
    root = 'Audio/LibriSpeech'
    segment = 3.
    interval = 1.5
    window.flip()
    # initialize Sound PTB engine. Sample rate has to be 44100/48000 to be compatible with the audio interface used.
    fs = 48000
    audio = SoundPTB(stereo=True, volume=1.0, sampleRate=48000, blockSize=128, hamming=False, autoLog=True)

    x_n = []
    if noise is not None:
        assert snr is not None
        _, x_n = read_audio('Audio/Noises/%s.wav' % noise, target_fs=48000)

    def stimulus(x1, x2, interval_duration, stim_fs=16000, stim_level=-10.0, stim_ear='left'):
        x1 = x1 / rms(x1) * 0.05
        x2 = x2 / rms(x2) * 0.05

        x1 = apply_raised_cosine_ramp(x1, stim_fs, onset_duration_ms=50, offset_duration_ms=50)
        x2 = apply_raised_cosine_ramp(x2, stim_fs, onset_duration_ms=50, offset_duration_ms=50)
        if np.random.randint(0, 1):
            x_prim = np.concatenate((x1, silence(interval_duration, stim_fs), x2), axis=0)
        else:
            x_prim = np.concatenate((x2, silence(interval_duration, stim_fs), x1), axis=0)

        if stim_ear == 'left':
            x_left, x_right = x_prim, np.zeros_like(x_prim)
        elif stim_ear == 'right':
            x_left, x_right = np.zeros_like(x_prim), x_prim
        else:
            x_left, x_right = x_prim, x_prim

        x_left = apply_raised_cosine_ramp(x_left, stim_fs, onset_duration_ms=20, offset_duration_ms=20)
        x_right = apply_raised_cosine_ramp(x_right, stim_fs, onset_duration_ms=20, offset_duration_ms=20)
        x_left = attenuate_sound(x_left, stim_level)
        x_right = attenuate_sound(x_right, stim_level)
        x_prim = [x_left, x_right]
        if np.amax(np.abs(x_prim)) >= 1:
            print(np.amax(np.abs(x_left)), np.amax(np.abs(x_right)))
            print('this will clip!')
        return np.array(x_prim).transpose()

    if participant == 'training':
        filepath = 'Audio/speaker_embeddings/training_clean.pkl'
    else:
        filepath = 'Audio/speaker_embeddings/speaker_sample_pairs_%s_%s.pkl' % (noise if noise is not None else 'clean',
                                                                          str(snr) if snr is not None else '')
    meta = pd.read_pickle(filepath).to_dict('records')

    text = visual.TextStim(window, text="", pos=(0, 0), color=(1, 1, 1))
    text.draw()
    window.flip()

    text.setText("Session %s\n%s %s\nPress space to begin" % (str(session+1) if participant != 'training' else
                                                              'Training', noise if noise is not None else '',
                                                              str(snr) if snr is not None else ''))

    text.draw()
    window.flip()
    key = event.waitKeys(keyList=['space', 'escape'])
    if 'escape' in key:
        return
    results = []
    for trial in range(session * n_trials, (session+1) * n_trials):
        text.setText("")
        text.draw()
        window.flip()
        core.wait(1.)
        _, speech1 = read_audio(os.path.join(root, meta[trial]['speech_id1'].replace('.wav', '.flac')), target_fs=fs)
        _, speech2 = read_audio(os.path.join(root, meta[trial]['speech_id2'].replace('.wav', '.flac')), target_fs=fs)
        speech1 = speech1[:int(fs * segment)]
        speech2 = speech2[:int(fs * segment)]
        if reverse_order:
            speech1, speech2 = speech2, speech1

        if noise is not None:
            # get random noise segment
            noise_start = np.random.randint(0, len(x_n) - segment * fs - 1)
            x_n_segment = x_n[noise_start:int(noise_start+segment*fs)]

            # apply the same noise in both segments in the same SNR.
            pre_snr = np.sqrt(np.mean(speech1 ** 2)) / (np.sqrt(np.mean(x_n_segment ** 2)) + 1e-6)
            scale_factor = 10. ** (-1 * snr / 20.) * pre_snr

            speech1 = speech1[:len(x_n)] + x_n_segment * scale_factor if len(speech1) > len(x_n_segment) \
                else speech1 + x_n[:len(speech1)] * scale_factor

            pre_snr = np.sqrt(np.mean(speech2 ** 2)) / (np.sqrt(np.mean(x_n_segment ** 2)) + 1e-6)
            scale_factor = 10. ** (-1 * snr / 20.) * pre_snr
            speech2 = speech2[:len(x_n)] + x_n_segment * scale_factor if len(speech2) > len(x_n_segment) \
                else speech2 + x_n[:len(speech2)] * scale_factor

        x = stimulus(speech1, speech2, interval, stim_fs=fs, stim_level=level, stim_ear=ear)

        audio.setSound(x)
        audio.play()
        core.wait(x.shape[0] / audio.sampleRate - 0.05)

        text.setText("Speaker\n Same  or  Different?\n  1    <->    0   ")
        text.draw()
        window.flip()
        event.clearEvents()
        key = event.waitKeys(keyList=['1', '0', 'return', 'escape'])
        if 'escape' in key:
            return

        if 'return' not in key:
            speaker_id1, speaker_id2 = int(meta[trial]['speech_id1'].split('-')[0]), int(
                meta[trial]['speech_id2'].split('-')[0])
            response = 1 * ('1' in key)
            is_correct = 'Correct' if (speaker_id1 == speaker_id2) == response else 'Wrong'
            if feedback:
                text.setText(is_correct)
            else:
                text.setText('Next\n\n%d/%d' % ((trial % n_trials)+1, n_trials))
            text.draw()
            window.flip()
            core.wait(1.)

            truth = 1 * (speaker_id1 == speaker_id2)
            model_prediction = meta[trial]['prediction']
            is_prediction_correct = 'Correct' if model_prediction == truth else 'Wrong'
            print('%d - %s - Response: %s - '
                  'Predicted: %s - '
                  'Similarity: %.2f - '
                  '[%s, %s]' % (trial, 'Same' if truth else 'Diff', is_correct, is_prediction_correct,
                                meta[trial]['similarity'], meta[trial]['speech_id1'], meta[trial]['speech_id2']))

            result = {'speech_id1': meta[trial]['speech_id1'],
                      'speech_id2': meta[trial]['speech_id2'],
                      'snr': snr,
                      'noise': noise,
                      'response': response,
                      'prediction': model_prediction,
                      'similarity': meta[trial]['similarity'],
                      'truth': 1 * (speaker_id1 == speaker_id2),
                      'session': session,
                      'ear': ear,
                      'participant': participant,
                      'trial': trial
                      }
            results.append(result)
        else:
            pd.DataFrame(results).to_pickle('Results/speaker/' + participant + '_' + str(session) + '_' +
                                            time.strftime("%h-%d-%Y_%I.%M.pkl"))
            break
        event.clearEvents()  # clear other (e.g., mouse) events - they clog the buffer
    if participant == 'training':
        pd.DataFrame(results).to_pickle('Results/speaker/training' + time.strftime("%h-%d-%Y_%I.%M.pkl"))
    else:
        pd.DataFrame(results).to_pickle('Results/speaker/' + participant + '_' + str(session) +
                                        time.strftime("%h-%d-%Y_%I.%M.pkl"))
    text.setText("Session End!\n\n:)")
    text.draw()
    window.flip()
    core.wait(1.)


def run_speaker_discrimination(participant):
    speaker_discrimination_block('training', session=0, n_trials=5, feedback=True)
    speaker_discrimination_block(participant, session=0)
    speaker_discrimination_block(participant, session=1)
    speaker_discrimination_block(participant, session=0, noise='iltass', snr=5)
    speaker_discrimination_block(participant, session=1, noise='iltass', snr=5)


window = None
if __name__ == '__main__':
    participant = input("Type your participant ID: ")
    check_data()
    window = visual.Window(fullscr=True, pos=[0, 0], color=(0, 0, 0))
    run_speaker_discrimination(participant)
    window.close()
    core.quit()
